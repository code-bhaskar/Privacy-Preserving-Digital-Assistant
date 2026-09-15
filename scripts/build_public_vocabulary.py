"""Rebuild the static public vocabulary used by the escalation DP mechanism.

`fl/public_vocabulary.txt` is the committed source of truth for the mechanism's
output space; this script only regenerates it, deliberately. Sources are strictly
public and already committed in this repository: this project's own documentation
plus a curated everyday-word supplement. No user record, prompt, note or training
example is ever read.

Regenerating matters: the vocabulary size `k` sets the k-RR retention probability
`e^eps / (e^eps + k - 1)`, so a doc edit that adds words slightly changes the
mechanism. The file therefore carries a SHA-256 digest of its own word list, and
`--check` verifies that digest (integrity of the committed asset), not agreement
with the current documentation.
"""

import hashlib

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "fl" / "public_vocabulary.txt"

DOCUMENT_SOURCES = [
    "README.md",
    ".env.example",
    "docs/LOCAL_LLM_INTENT.md",
    "docs/LOCAL_SUMMARIES.md",
    "docs/NOTIFICATIONS_AND_COMMANDS.md",
    "docs/PRD.md",
    "docs/ROBOT_ASSETS.md",
    "docs/SECURITY_AND_DP.md",
    "models/snips/README.md",
]

# Everyday question vocabulary, written by hand so that ordinary out-of-scope
# requests keep enough meaning after perturbation. Public common knowledge.
SUPPLEMENT = """
about above across after again against age ago air all almost alone along
already also although always am among an and angry animal another answer any
anyone anything are around art as ask at away baby back bad bag ball bank bar
base be beach bear beat beautiful because become bed before begin behind being
below best better between big bike bird birth bit black blue board boat body
book both bottle box boy bread break bright bring brother brown build bus
business but buy by cake call came can candle candy car card care carry case
cat catch cause center certain chair chance change cheap check chicken child
choice choose city clean clear close clothes cloud cold color come common
company computer cook cool could country course cover cow cross cup cut dance
dark date daughter day dead decide deep desk did die different dinner direction
dirty do doctor does dog dollar done door down draw dream dress drink drive
drop dry during each ear early earth east easy eat edge egg eight either else
empty end enough enter even evening ever every example except exercise eye
face fact fall family far farm fast father fear feed feel feet fell felt few
field fight fill film final find fine finger finish fire first fish fit five
fix floor flower fly follow food foot for force forget form forward found four
free fresh friend from front fruit full fun game garden gas gate gave general
gentle get gift girl give glass go gold gone good got government grass great
green grew ground group grow guess hair half hall hand happen happy hard hat
hate have he head hear heart heat heavy held hello help her here hers hey hi
hide high hill him his hit hold hole holiday home hope horse hospital hot hour
house how however huge human hundred hunger hurt husband ice idea if ill image
imagine in inch include income increase indeed inside instead into iron is
island it its job join journey joy jump keep kept key kick kid kind king
kitchen knee knew knife knock know known lady lake lamp land language large
last late laugh law lay lead learn least leave left leg less let letter level
lie life lift light like line lion list listen little live long look lose lot
loud love low luck lunch machine made magic mail main make man many map mark
market marry match matter may maybe me meal mean measure meat medical meet
member memory men mention method middle might mile milk million mind minute
mirror miss moment money month moon more morning most mother mountain mouth
move much music must my name narrow nation nature near necessary neck need
never new news next nice night nine no noise none nor north nose not note
nothing notice now number nurse object ocean of off offer office often oh oil
old on once one only open or orange order other our out outside over own page
pain paint pair paper parent park part party pass past path pay peace pen
people perfect perhaps person pet phone photo picture piece place plan plant
plate play please pocket point police poor population position possible post
pound power practice prepare present press pretty price print probably problem
produce promise proper protect pull push put question quick quiet quite race
radio rain raise ran rather reach read ready real reason receive record red
remain remember remove repair repeat reply report rest result return rich ride
right ring rise river road rock roll room round row rule run sad safe said sail
salt same sand sat save saw say scale school science sea season seat second see
seed seem seen sell send sense sent sentence separate serve seven several shall
shape share sharp she sheet ship shoe shop short should shoulder shout show
shut sick side sight sign silence simple since sing single sister sit six size
skin sky sleep slow small smell smile smoke snow so soft soil soldier some son
song soon sorry sort sound south space speak special speed spell spend sport
spring square stand star start state station stay step stick still stone stood
stop store storm story straight strange street strong study such sudden sugar
summer sun supply support sure surface surprise sweet swim system table tail
take talk tall teach team tell ten term test than thank that the their them
then there these they thick thin thing think third this those though thought
thousand three through throw thus ticket tie till time tiny tired to today
together told tomorrow tone too took tooth top total touch toward town track
trade train travel tree trip trouble true try turn twelve twenty two type ugly
uncle under understand unit until up upon us use usual valley value various
very view village visit voice wait wake walk wall want war warm was wash waste
watch water wave way we wear weather week weight welcome well went were west
wet what wheel when where whether which while white who whole whom whose why
wide wife wild will win wind window wing winter wire wise wish with within
without woman wonder wood word work world worry worst worth would write wrong
yard year yellow yes yesterday yet you young your youth
""".split()

# Second supplement: everyday request vocabulary (places, errands, study,
# technology, food) so ordinary out-of-scope questions stay intelligible.
SUPPLEMENT_REQUESTS = """
advice address afternoon airport algorithm appointment article artist author
autumn baby banana bank battery beach birthday bill biology book breakfast bus
business butter camera campus capital career charge chemistry city clinic
coffee college concert computer cost cricket currency customer dance deadline
dentist department dessert dictionary diet dinner discount doctor download
electricity email engineer english envelope exam exercise family festival
flight french football furniture garden german gift grammar grocery gym history
hindi hospital hotel interview java journey laptop library license lunch
machine magazine market mathematics medicine meeting message mobile morning
museum music network news novel nurse office opinion package passport password
physics poem presentation product professor program programming project python
quality queue receipt recipe rent report research restaurant result resume rice
salary school science season security shop software solution spanish station
student study supermarket surgery syllabus teacher technology telephone
television temperature ticket tomato topic train translate travel university
vacation vegetable visa vocabulary wallet weather website wedding
weekend window
""".split()


def build():
    words = set()
    for relative in DOCUMENT_SOURCES:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8").lower()
        words.update(re.findall(r"[a-z]{2,14}", text))
    words.update(SUPPLEMENT)
    words.update(SUPPLEMENT_REQUESTS)
    # Contractions and code fragments are not useful mechanism output symbols.
    words = {w for w in words if w.isalpha()}
    return sorted(words)


def digest(words):
    return hashlib.sha256("\n".join(words).encode()).hexdigest()


def render(words):
    return (
        "# Public output vocabulary for the escalation differential-privacy mechanism.\n"
        "# Committed source of truth for the k-RR output space; do not hand-edit.\n"
        "# Regenerate deliberately with: python scripts/build_public_vocabulary.py\n"
        "# Derived only from this repository's public documentation plus a curated\n"
        "# everyday-word supplement. Contains no user data of any kind.\n"
        f"# words: {len(words)}  sha256: {digest(words)}\n"
        + "\n".join(words)
        + "\n"
    )


def committed():
    lines = OUTPUT.read_text(encoding="utf-8").splitlines()
    words = [line for line in lines if line and not line.startswith("#")]
    stated = next(
        (line.split("sha256:")[1].strip() for line in lines if "sha256:" in line), ""
    )
    return words, stated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify the committed asset, do not write"
    )
    arguments = parser.parse_args()
    if arguments.check:
        words, stated = committed()
        actual = digest(words)
        if actual != stated:
            print(f"MISMATCH: header says {stated}, content hashes to {actual}")
            raise SystemExit(1)
        print(f"OK: {len(words)} committed vocabulary words, sha256 {actual[:16]}…")
        return
    words = build()
    OUTPUT.write_text(render(words), encoding="utf-8")
    print(
        f"Wrote {len(words)} public vocabulary words to {OUTPUT.relative_to(ROOT)} "
        f"(sha256 {digest(words)[:16]}…)"
    )


if __name__ == "__main__":
    main()
