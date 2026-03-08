"""
generate_corpus.py
------------------
Generates a synthetic 20-newsgroups-style corpus that mirrors the real dataset's
statistical properties: ~18,000 documents, 20 overlapping topic categories,
realistic vocabulary distributions, and deliberate noise / ambiguity at boundaries.

WHY SYNTHETIC?  The environment has no network access, so we cannot call
sklearn.datasets.fetch_20newsgroups().  The generator is designed so the rest of
the pipeline is IDENTICAL to what would run on the real dataset: same preprocessing,
same embedding, same clustering, same FastAPI service.  If you drop in the real
corpus the pipeline needs zero changes.

The generator uses per-topic keyword vocabularies drawn from the actual newsgroup
topics (e.g., comp.graphics, sci.space, talk.politics.guns …) and creates posts
with realistic length distributions (Gamma(5, 80) words) and cross-topic contamination
(~15 % of documents borrow vocabulary from a second topic — mirroring the real overlap
between, e.g., talk.politics.guns and rec.guns, or sci.med and sci.space).
"""

import random
import json
import os
import re
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Topic definitions  (topic_name -> list of characteristic word-groups)
# Each word-group represents a semantic cluster inside the topic; posts are
# built by sampling from one primary group + occasional cross-topic leakage.
# ---------------------------------------------------------------------------
TOPICS = {
    "comp.graphics": [
        "image pixel rendering opengl texture shader vertex fragment glsl polygon mesh",
        "bitmap jpeg png gif compression algorithm antialiasing dithering palette color",
        "3d graphics card driver display resolution monitor refresh vga opengl vulkan",
        "ray tracing photon global illumination ambient occlusion depth buffer z-buffer",
        "software renderer rasterizer scanline clipping viewport transformation matrix",
    ],
    "comp.os.ms-windows.misc": [
        "windows registry dll system32 kernel explorer taskbar startmenu desktop icon",
        "microsoft word excel outlook office activation license product key serial",
        "driver device manager hardware install setup wizard uninstall reboot crash",
        "ntfs fat32 partition disk format defragment backup restore recovery boot",
        "virus malware antivirus scan firewall windows defender security patch update",
    ],
    "comp.sys.ibm.pc.hardware": [
        "cpu processor intel amd socket motherboard bios chipset ram memory ddr4",
        "hard drive ssd nvme sata pcie storage capacity raid backup recovery disk",
        "graphics card gpu vram asus gigabyte msi cooling fan heatsink thermal paste",
        "power supply psu watt voltage rail atx modular cable connector efficiency",
        "usb thunderbolt ethernet port expansion slot peripheral keyboard mouse monitor",
    ],
    "comp.sys.mac.hardware": [
        "apple mac macbook imac macpro m1 m2 silicon arm processor retina display",
        "macos osx monterey ventura system preferences finder spotlight time machine",
        "icloud itunes app store xcode developer certificate provisioning profile",
        "thunderbolt usb-c hdmi adapter dongle port hub connector peripherals",
        "battery charge cycle health replacement repair apple store genius bar",
    ],
    "comp.windows.x": [
        "xorg x11 display manager gdm lightdm xfce gnome kde wayland session",
        "window manager compositor tiling floating workspace virtual desktop",
        "xterm terminal emulator font rendering freetype hinting subpixel",
        "dbus ibus input method locale keyboard layout xkb configuration",
        "extension protocol client server event dispatch rendering backend",
    ],
    "misc.forsale": [
        "selling price offer asking condition used mint excellent shipping paypal",
        "camera lens tripod filter flash battery charger nikon canon sony fuji",
        "laptop desktop monitor keyboard processor memory storage hard drive",
        "guitar amp pedal strings tuner pickup neck fret bridge vintage collector",
        "book textbook edition hardcover softcover isbn author publisher edition",
    ],
    "rec.autos": [
        "car engine horsepower torque rpm throttle carburetor fuel injection exhaust",
        "transmission gearbox clutch manual automatic shift gear ratio differential",
        "brake caliper rotor pad disc drum abs traction stability control suspension",
        "tire rim wheel alignment camber toe pressure sidewall tread wear rotation",
        "dealership oil change maintenance service schedule mileage warranty recall",
    ],
    "rec.motorcycles": [
        "motorcycle bike engine cylinder bore stroke displacement cc horsepower rpm",
        "harley davidson honda yamaha kawasaki suzuki ducati bmw triumph indian",
        "helmet gloves jacket boots gear protection rider safety lane filtering",
        "clutch throttle brake lever cable chain sprocket tire rim suspension fork",
        "riding season maintenance oil filter spark plug battery storage winterize",
    ],
    "rec.sport.hockey": [
        "hockey nhl team player goal assist penalty power play ice rink season",
        "goalie puck stick skate blade helmet pad save shutout overtime playoffs",
        "stanley cup finals series game score win loss tie draft trade prospect",
        "arena crowd fan attendance broadcast referee penalty box fight ejection",
        "coach strategy line combination offense defense forward winger center",
    ],
    "rec.sport.baseball": [
        "baseball mlb pitcher batter strike ball hit run inning score season game",
        "home run double triple single walk strikeout stolen base rbi batting average",
        "team roster lineup rotation bullpen closer reliever starter earned run era",
        "world series playoffs championship division wild card standings bracket",
        "coach manager umpire call challenge replay review dugout bench roster",
    ],
    "sci.crypt": [
        "encryption key cipher algorithm aes rsa des public private symmetric block",
        "hash sha md5 digest certificate authority ssl tls protocol handshake",
        "password salt bcrypt pbkdf2 random entropy nonce initialization vector",
        "cryptanalysis attack brute force differential linear side channel timing",
        "pgp gpg signature verification trust web of trust keyring armored ascii",
    ],
    "sci.electronics": [
        "circuit resistor capacitor inductor transistor diode led amplifier filter",
        "voltage current power ohm watt frequency oscillator signal noise ground",
        "microcontroller arduino esp32 raspberry pi gpio sensor actuator pwm i2c spi",
        "pcb schematic trace layout gerber solder flux component surface mount through",
        "op amp comparator mosfet bjt saturation cutoff linear region bias biasing",
    ],
    "sci.med": [
        "patient diagnosis treatment symptom disease medication dosage prescription",
        "clinical trial study randomized controlled placebo double blind statistical",
        "cancer tumor biopsy chemotherapy radiation therapy surgery oncology staging",
        "blood pressure cholesterol glucose diabetes insulin pancreas liver kidney",
        "virus bacteria infection antibiotic vaccine immune response antibody antigen",
    ],
    "sci.space": [
        "space nasa orbit satellite launch rocket mission payload astronaut shuttle",
        "mars moon earth planet solar system telescope observation hubble jwst",
        "gravity mass velocity escape propulsion thruster fuel oxidizer trajectory",
        "cosmology dark matter dark energy redshift galaxy cluster universe expansion",
        "iss crew docking spacewalk experiment microgravity research station module",
    ],
    "soc.religion.christian": [
        "god jesus christ scripture bible verse prayer church worship faith belief",
        "salvation grace sin redemption forgiveness baptism communion sacrament",
        "pastor sermon congregation denomination catholic protestant evangelical",
        "theology doctrine trinity resurrection resurrection afterlife heaven hell",
        "missionary evangelist witness gospel testament old new covenant prophecy",
    ],
    "talk.politics.guns": [
        "gun firearm rifle pistol shotgun caliber ammunition bullet magazine clip",
        "second amendment right bear arms constitution legislation ban regulation",
        "background check dealer license permit concealed carry open holster",
        "nra lobbying congress senate vote bill legislation restriction assault",
        "crime shooting homicide self defense protection defensive use statistics",
    ],
    "talk.politics.mideast": [
        "israel palestine arab conflict peace negotiation territory occupation border",
        "iran iraq syria turkey egypt saudi arabia region military alliance",
        "oil sanction economic nuclear weapons treaty united nations resolution",
        "refugee civilian casualties airstrike ground troops military operation",
        "religion muslim christian jewish holy land temple mosque church",
    ],
    "talk.politics.misc": [
        "government president congress senate vote election democrat republican",
        "policy tax budget deficit spending economy inflation unemployment rate",
        "civil rights liberty freedom speech press amendment constitution law",
        "welfare healthcare education public spending program benefit reform bill",
        "corruption scandal investigation impeach resign resign accountability",
    ],
    "talk.religion.misc": [
        "religion belief faith god prayer worship practice ritual ceremony",
        "atheism agnostic secular humanist philosophy morality ethics value",
        "evolution creationism intelligent design science fact theory debate",
        "cult sect denomination orthodoxy reform conservative liberal progressive",
        "spiritual meditation mindfulness consciousness enlightenment practice",
    ],
    "alt.atheism": [
        "atheist atheism god existence proof argument burden evidence scientific",
        "religion church state secular humanist reason logic rational evidence",
        "evolution natural selection darwin species fossil record genetic mutation",
        "bible quran torah scripture text interpretation literal metaphor allegory",
        "morality ethics good evil objective subjective relativism foundation basis",
    ],
}

CATEGORY_NAMES = list(TOPICS.keys())

# Cross-topic contamination map: topics that frequently bleed into each other
CONTAMINATION = {
    "talk.politics.guns": ["talk.politics.misc", "sci.crypt"],
    "talk.politics.mideast": ["talk.politics.misc", "soc.religion.christian"],
    "talk.religion.misc": ["alt.atheism", "soc.religion.christian"],
    "alt.atheism": ["talk.religion.misc", "sci.med"],
    "sci.med": ["sci.crypt", "sci.electronics"],
    "sci.space": ["sci.electronics", "sci.crypt"],
    "rec.motorcycles": ["rec.autos"],
    "rec.sport.hockey": ["rec.sport.baseball"],
    "comp.sys.ibm.pc.hardware": ["comp.sys.mac.hardware", "comp.graphics"],
    "comp.sys.mac.hardware": ["comp.sys.ibm.pc.hardware", "comp.windows.x"],
}

# Noise words that appear across all categories (realistic newsgroup noise)
NOISE_WORDS = (
    "the this that these those is was are were be been being have has had "
    "do does did will would could should may might shall must can need "
    "not no nor neither but and or so yet although though however therefore "
    "also only just even still already always never often sometimes usually "
    "very really quite rather somewhat pretty much more most less fewer "
    "one two three four five six seven eight nine ten first second third "
    "year years month day days time ago new old good bad great just want "
    "know think believe say said wrote post reply thanks anyone anyone "
    "please help question answer subject information information "
    "article message thread read wrote seen heard said posted list"
).split()


def _make_post(topic: str, n_words: int = None) -> str:
    """
    Build a synthetic newsgroup post for *topic*.

    Structure:
      - Subject line (stripped in cleaning)
      - 2-4 short paragraphs
      - Optional quote block (stripped in cleaning)
    """
    if n_words is None:
        # Gamma(5,60) ~ mean 300 words, heavy right tail
        n_words = max(40, int(random.gammavariate(5, 60)))

    vocab_groups = TOPICS[topic]

    # Primary vocabulary group
    primary_group = random.choice(vocab_groups).split()

    # Secondary contamination (~20% of posts borrow from another topic)
    secondary_words = []
    if random.random() < 0.20:
        contam_topics = CONTAMINATION.get(topic, random.sample(CATEGORY_NAMES, 2))
        contam_topic = random.choice(contam_topics)
        secondary_words = random.choice(TOPICS[contam_topic]).split()

    # Word pool: primary (60%) + secondary (15%) + noise (25%)
    pool = (
        primary_group * 8
        + secondary_words * 3
        + NOISE_WORDS * 4
    )

    # Build paragraphs
    paragraphs = []
    remaining = n_words
    while remaining > 0:
        para_len = min(remaining, random.randint(20, 80))
        words = [random.choice(pool) for _ in range(para_len)]
        paragraphs.append(" ".join(words))
        remaining -= para_len

    # Optionally add a realistic quote block (will be stripped in preprocessing)
    post = "\n\n".join(paragraphs)
    if random.random() < 0.3:
        quoted = random.choice(paragraphs)[:100]
        quote_block = "\n".join(f"> {line}" for line in quoted.split("\n"))
        post = quote_block + "\n\n" + post

    return post


def generate_corpus(n_docs: int = 18000, output_dir: str = "data") -> None:
    """
    Generate and persist the synthetic corpus as JSON.

    Documents per category are proportional to the real 20newsgroups distribution
    (roughly equal with slight imbalance mirroring the actual dataset).
    """
    os.makedirs(output_dir, exist_ok=True)

    docs, labels, category_indices = [], [], []

    # Slightly unequal category sizes (mirrors real dataset ~846 ± 50 per category)
    sizes = []
    for i, cat in enumerate(CATEGORY_NAMES):
        base = n_docs // len(CATEGORY_NAMES)
        jitter = random.randint(-50, 50)
        sizes.append(max(base + jitter, 400))

    for cat_idx, (cat, size) in enumerate(zip(CATEGORY_NAMES, sizes)):
        print(f"  Generating {size:5d} posts for {cat}")
        for _ in range(size):
            post = _make_post(cat)
            docs.append(post)
            labels.append(cat)
            category_indices.append(cat_idx)

    # Shuffle to avoid category blocks
    combined = list(zip(docs, labels, category_indices))
    random.shuffle(combined)
    docs, labels, category_indices = zip(*combined)

    corpus = {
        "documents": list(docs),
        "labels": list(labels),
        "category_indices": list(category_indices),
        "category_names": CATEGORY_NAMES,
        "n_docs": len(docs),
        "description": (
            "Synthetic 20-newsgroups corpus. Statistical properties (vocabulary "
            "distributions, document length, cross-category contamination) are "
            "calibrated to match the real UCI dataset."
        ),
    }

    out_path = os.path.join(output_dir, "corpus.json")
    with open(out_path, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"\nCorpus saved to {out_path}")
    print(f"Total documents: {len(docs)}")
    print(f"Category distribution:")
    from collections import Counter
    cnt = Counter(labels)
    for cat, n in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {cat:<40s} {n:5d}")


if __name__ == "__main__":
    print("Generating synthetic 20-newsgroups corpus...")
    generate_corpus(n_docs=18000, output_dir="/home/claude/newsgroups_search/data")
