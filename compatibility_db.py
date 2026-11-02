"""
Compatibility Database + Search Model (offline, no external APIs)

* Reads the AMXT sheet of the ISV compatibility workbook.
* Normalises status values and extracts PTF / APAR IDs from the free-text cells.
* "Trains" a scikit-learn TF-IDF index (character + word n-grams) on
  Vendor + Product + Version so products can be found even with typos or
  partial names (e.g. "db2 13", "endevr", "file aid").
* Builds search links (web, vendor-site, official APAR page / vendor portal)
  for every PTF / APAR ID found in the sheet.

Only scikit-learn / pandas / joblib are needed -> safe for IBM sites with no
outbound AI services.
"""
from __future__ import annotations

import glob
import os
import re
from urllib.parse import quote_plus

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHEET = "AMXT"
MODEL_FILE = os.path.join(BASE_DIR, "compat_model.joblib")

C_COMPAT = "R3.2 Compatible (Y)Yes or (N)No"
C_PTF = "If Yes, PTF Required?"
C_LINK = "Link to verify compatibility report"
C_MSG = "Messsage ID to identify Product version"
C_LOGIN = "Information Available without login"
C_COMMENTS = "Comments/Link"

STATUSES = ["Compatible", "Not compatible", "No info", "Unverified"]
DEFAULT_RELEASE = "3.2"          # the only release the sheet actually assesses


# ----------------------------------------------------------------- helpers
def find_excel(path: str | None = None) -> str:
    names = [path] if path else []
    names += ["ISV Compatibility assessment -Data.xlsx",
              "ISV_Compatibility_assessment_-Data.xlsx"]
    for n in names:
        if not n:
            continue
        for p in (n, os.path.join(BASE_DIR, n)):
            if os.path.exists(p):
                return os.path.abspath(p)
    hits = sorted(glob.glob(os.path.join(BASE_DIR, "ISV*.xlsx")))
    if hits:
        return hits[0]
    raise FileNotFoundError("ISV Compatibility assessment Excel file not found next to the app.")


def _txt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return ""
    return re.sub(r"\s+", " ", str(v).replace("\xa0", " ")).strip()


# PTF / APAR / SYSMOD ids: 7 chars, 2-3 letters + 3-5 digits (+ optional letter)
ID_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Z0-9]{7}(?![A-Za-z0-9]))[A-Z]{2,3}\d{3,5}[A-Z]?(?![A-Za-z0-9])"
)
APAR_PREFIXES = ("OA", "PH", "PI", "PM")
# "z/OS 3.2", "z/OS V3R2", "z/OS.V3R1", "Zos 2.3"
RELEASE_RE = re.compile(r"z/?os[\s.\-]*v?(\d)\s*[r.]?\s*(\d)", re.I)
URL_RE = re.compile(r"https?://[^\s,;)]+", re.I)


def detect_releases(text: str) -> set[str]:
    return {f"{a}.{b}" for a, b in RELEASE_RE.findall(text or "")}


def id_kind(pid: str) -> str:
    return "APAR" if pid.startswith(APAR_PREFIXES) else "PTF"


def norm_status(v) -> str:
    s = _txt(v).lower()
    if not s:
        return "Unverified"
    if s.startswith("yes") or s == "y":
        return "Compatible"
    if s.startswith("no info"):
        return "No info"
    if s.startswith("no"):
        return "Not compatible"
    return "Unverified"          # e.g. "same as line no 25,26"


def extract_ids(sources: list[tuple[str, str]]) -> list[dict]:
    found: dict = {}
    for label, text in sources:
        if not text:
            continue
        rels = detect_releases(text)
        inferred = not rels
        if inferred:
            rels = {DEFAULT_RELEASE}      # cell has no release text -> sheet default
        for m in ID_RE.finditer(text):
            pid = m.group(0)
            found[(pid, label)] = dict(id=pid, kind=id_kind(pid), source=label,
                                       releases=sorted(rels), inferred=inferred,
                                       evidence=text)
    return list(found.values())


# ----------------------------------------------------------------- links
VENDOR_SITES = {
    "bmc": (("docs.bmc.com", "support.bmc.com"), "https://support.bmc.com"),
    "broadcom": (("support.broadcom.com", "techdocs.broadcom.com"), "https://support.broadcom.com"),
    "ibm": (("ibm.com",), "https://www.ibm.com/support/fixcentral/"),
    "rocket": (("docs.rocketsoftware.com", "my.rocketsoftware.com"), "https://my.rocketsoftware.com"),
    "precisely": (("support.precisely.com",), "https://support.precisely.com"),
}


def _vendor_key(vendor: str) -> str | None:
    v = vendor.lower()
    return next((k for k in VENDOR_SITES if k in v), None)


def build_links(pid: str, kind: str, vendor: str) -> dict:
    google = "https://www.google.com/search?q="
    out = {"web": google + quote_plus(f'"{pid}" {kind} z/OS'),
           "vendor_search": "", "official": ""}
    key = _vendor_key(vendor)
    if key:
        sites = " OR ".join(f"site:{s}" for s in VENDOR_SITES[key][0])
        out["vendor_search"] = google + quote_plus(f'"{pid}" ({sites})')
        out["official"] = VENDOR_SITES[key][1]
    if kind == "APAR":                       # IBM APAR pages have a stable URL pattern
        out["official"] = f"https://www.ibm.com/support/pages/apar/{pid}"
    return out


# ----------------------------------------------------------------- data frame
def build_frame(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=SHEET)
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.dropna(how="all", subset=["Vendor", "Product"]).reset_index(drop=True)

    rows = []
    for _, r in raw.iterrows():
        ptf, com, msg = _txt(r.get(C_PTF)), _txt(r.get(C_COMMENTS)), _txt(r.get(C_MSG))
        link_cell = _txt(r.get(C_LINK))
        urls = URL_RE.findall(link_cell)
        login = _txt(r.get(C_LOGIN))
        rows.append(dict(
            vendor=_txt(r.get("Vendor")) or "Unknown",
            product=_txt(r.get("Product")),
            version=_txt(r.get("Version")) or "-",
            status=norm_status(r.get(C_COMPAT)),
            status_raw=_txt(r.get(C_COMPAT)),
            ptf_text=ptf, comments=com, msgid=msg,
            link=urls[0] if urls else "",
            link_hint="" if urls else link_cell,
            login_free={"yes": "Yes", "no": "No"}.get(login.lower(), "Unknown"),
            ids=extract_ids([("PTF column", ptf), ("Comments", com), ("Message ID", msg)]),
        ))
    df = pd.DataFrame(rows)
    df["label"] = df["vendor"] + " | " + df["product"] + " | " + df["version"]
    df["n_ids"] = df["ids"].map(len)
    df["id_list"] = df["ids"].map(lambda L: ", ".join(sorted({i["id"] for i in L})))
    df["all_releases"] = df["ids"].map(lambda L: sorted({x for i in L for x in i["releases"]}))
    df["mentioned_releases"] = (df["ptf_text"] + " " + df["comments"] + " " + df["msgid"]).map(
        lambda t: sorted(detect_releases(t)))
    return df


def available_releases(df: pd.DataFrame) -> list[str]:
    rels = {DEFAULT_RELEASE}
    for L in df["mentioned_releases"]:
        rels.update(L)
    return sorted(rels, key=lambda r: tuple(int(x) for x in r.split(".")), reverse=True)


def status_for(row, release: str) -> str:
    return row["status"] if release == DEFAULT_RELEASE else "Not assessed"


def release_evidence(row, release: str) -> list[tuple[str, str]]:
    out = []
    for label, text in (("PTF column", row["ptf_text"]), ("Comments", row["comments"]),
                        ("Message ID", row["msgid"])):
        if text and release in detect_releases(text):
            out.append((label, text))
    return out


def ptf_records(df: pd.DataFrame, release: str | None) -> pd.DataFrame:
    """One row per PTF/APAR id. release=None -> every release mentioned."""
    recs = []
    for _, r in df.iterrows():
        for it in r["ids"]:
            if release and release not in it["releases"]:
                continue
            lk = build_links(it["id"], it["kind"], r["vendor"])
            recs.append({
                "Vendor": r["vendor"], "Product": r["product"], "Version": r["version"],
                "ID": it["id"], "Type": it["kind"], "Source": it["source"],
                "Applies to z/OS": ", ".join(it["releases"]) + (" (sheet default)" if it["inferred"] else ""),
                "Evidence": it["evidence"],
                "Web search": lk["web"], "Vendor-site search": lk["vendor_search"] or None,
                "Official page / portal": lk["official"] or None,
                "Verification report": r["link"] or None,
            })
    cols = ["Vendor", "Product", "Version", "ID", "Type", "Source", "Applies to z/OS", "Evidence",
            "Web search", "Vendor-site search", "Official page / portal", "Verification report"]
    return pd.DataFrame(recs, columns=cols)


# ----------------------------------------------------------------- the "trained" model
class CompatModel:
    """TF-IDF retrieval model trained on the Excel sheet."""

    def __init__(self, df, char_vec, word_vec, Xc, Xw, docs, signature):
        self.df, self.char_vec, self.word_vec = df, char_vec, word_vec
        self.Xc, self.Xw, self.docs, self.signature = Xc, Xw, docs, signature

    @classmethod
    def train(cls, path: str) -> "CompatModel":
        df = build_frame(path)
        docs = (df["vendor"] + " " + df["product"] + " " + df["version"]).str.lower().tolist()
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        Xc, Xw = char_vec.fit_transform(docs), word_vec.fit_transform(docs)
        return cls(df, char_vec, word_vec, Xc, Xw, docs, _signature(path))

    def search(self, query: str, k: int = 8, min_score: float = 0.08) -> list[tuple[int, float]]:
        q = query.lower().strip()
        if not q:
            return []
        s = (0.6 * linear_kernel(self.char_vec.transform([q]), self.Xc)[0]
             + 0.4 * linear_kernel(self.word_vec.transform([q]), self.Xw)[0])
        s = s + 0.3 * np.array([q in d for d in self.docs])      # exact-substring boost
        order = np.argsort(-s)[:k]
        return [(int(i), float(min(s[i], 1.0))) for i in order if s[i] >= min_score]

    def similar(self, idx: int, k: int = 5) -> list[int]:
        s = linear_kernel(self.Xc[idx], self.Xc)[0]
        prod = self.df.loc[idx, "product"]
        order = [i for i in np.argsort(-s) if self.df.loc[i, "product"] != prod]
        return [int(i) for i in order[:k]]


def _signature(path: str):
    st = os.stat(path)
    return (round(st.st_mtime), st.st_size)


def load_or_train(path: str | None = None, force: bool = False) -> CompatModel:
    path = find_excel(path)
    if not force and os.path.exists(MODEL_FILE):
        try:
            m = joblib.load(MODEL_FILE)
            if m.signature == _signature(path):
                return m
        except Exception:
            pass                       # stale / different sklearn version -> retrain
    m = CompatModel.train(path)
    try:
        joblib.dump(m, MODEL_FILE)
    except Exception:
        pass                           # read-only deployment folder is fine
    return m
