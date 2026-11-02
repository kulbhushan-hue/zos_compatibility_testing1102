"""
z/OS Product Compatibility Dashboard (Streamlit) - interactive version
Pick a product + z/OS release  ->  PTFs / APARs with links to find them.
The search model is trained offline from the Excel sheet (scikit-learn).
"""
import pandas as pd
import streamlit as st

import compatibility_db as cdb

import inspect
_STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters
            else {"use_container_width": True})

st.set_page_config(page_title="z/OS Product Compatibility", page_icon="🖥️", layout="wide")

BADGE = {"Compatible": "🟢", "Not compatible": "🔴", "No info": "⚪",
         "Unverified": "🟡", "Not assessed": "⚫"}
LINK_CFG = {
    "Web search": st.column_config.LinkColumn("Web search", display_text="Search ↗"),
    "Vendor-site search": st.column_config.LinkColumn("Vendor-site search", display_text="Vendor docs ↗"),
    "Official page / portal": st.column_config.LinkColumn("Official page / portal", display_text="Open ↗"),
    "Verification report": st.column_config.LinkColumn("Verification report", display_text="Report ↗"),
}


@st.cache_resource(show_spinner="Training search model from the Excel sheet...")
def get_model():
    return cdb.load_or_train()


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


model = get_model()
df = model.df
RELEASES = cdb.available_releases(df)
VENDORS = sorted(df["vendor"].unique())

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("z/OS release")
    release = st.selectbox(
        "Release to check", RELEASES, index=RELEASES.index(cdb.DEFAULT_RELEASE),
        format_func=lambda r: f"z/OS {r}" + ("  (assessed in sheet)" if r == cdb.DEFAULT_RELEASE else "  (mentions only)"),
    )
    if release != cdb.DEFAULT_RELEASE:
        st.caption("The sheet only has a compatibility column for z/OS 3.2. For other releases "
                   "the app shows only PTF/APAR text that explicitly mentions that release.")
    st.divider()
    st.header("Filters (dashboard / worklist / inventory)")
    sel_vendors = st.multiselect("Vendors", VENDORS, default=VENDORS)
    st.divider()
    st.caption(f"Model: TF-IDF index over {len(df)} product/version rows, "
               f"{int((df['n_ids'] > 0).sum())} rows with PTF/APAR IDs.")
    if st.button("🔄 Retrain model from Excel"):
        st.cache_resource.clear()
        cdb.load_or_train(force=True)
        st.rerun()

fdf = df[df["vendor"].isin(sel_vendors)]

st.title("🖥️ z/OS ISV Product Compatibility Dashboard")
st.caption(f"Source: {cdb.find_excel().split('/')[-1]} (sheet {cdb.SHEET})  |  checking z/OS {release}")

tab_lookup, tab_plan, tab_dash, tab_ptf, tab_full = st.tabs([
    "🎯 Product Lookup", "🧰 Upgrade Planner", "📊 Dashboard",
    "🔧 PTF / APAR Worklist", "🔍 Full Inventory"])


# ------------------------------------------------------------------ result renderer
def render_row(idx: int, expanded: bool):
    row = df.loc[idx]
    status = cdb.status_for(row, release)
    title = f"{BADGE[status]} {row['vendor']} - {row['product']} {row['version']}"
    with st.expander(title, expanded=expanded):
        recs = cdb.ptf_records(df.loc[[idx]], release)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Status (z/OS {release})", status)
        m2.metric("PTF / APAR IDs", len(recs))
        m3.metric("Details without login", row["login_free"])
        m4.metric("Sheet raw answer", row["status_raw"] or "(blank)")

        if release != cdb.DEFAULT_RELEASE:
            ev = cdb.release_evidence(row, release)
            st.info(f"The sheet has no dedicated z/OS {release} assessment for this product. "
                    f"{'Text mentioning z/OS ' + release + ' is shown below.' if ev else 'No text in the sheet mentions z/OS ' + release + ' for it.'}")
            for src, txt in ev:
                st.write(f"**{src}:** {txt}")

        if recs.empty:
            other = sorted({x for i in row["ids"] for x in i["releases"]})
            msg = f"No PTF/APAR IDs recorded for z/OS {release}."
            if other:
                msg += f" IDs in this row relate to z/OS {', '.join(other)} (change the release in the sidebar)."
            st.warning(msg)
        else:
            st.markdown("**PTFs / APARs and where to find them**")
            st.dataframe(recs.drop(columns=["Vendor", "Product", "Version"]), hide_index=True,
                         **_STRETCH, column_config=LINK_CFG)
            st.download_button("⬇️ Download as CSV", csv_bytes(recs),
                               file_name=f"ptfs_{row['product'].replace(' ', '_')}_{row['version']}_zos{release}.csv",
                               key=f"dl_{idx}_{release}")

        st.markdown("**Verification report**")
        if row["link"]:
            st.link_button("Open vendor compatibility report ↗", row["link"], key=f"rep_{idx}")
        elif row["link_hint"]:
            q = "https://www.google.com/search?q=" + row["link_hint"].replace(" ", "+")
            st.write(f"No direct URL in the sheet. Hint: _{row['link_hint']}_")
            st.link_button("Search for it ↗", q, key=f"hint_{idx}")
        else:
            st.write("No report link in the sheet.")

        with st.expander("Raw notes from the sheet"):
            st.write(f"**PTF column:** {row['ptf_text'] or '-'}")
            st.write(f"**Comments:** {row['comments'] or '-'}")
            st.write(f"**Message ID:** {row['msgid'] or '-'}")

        sim = model.similar(idx, 4)
        if sim:
            st.caption("Similar products: " + "  •  ".join(df.loc[i, "label"] for i in sim))


# ------------------------------------------------------------------ TAB 1: lookup
with tab_lookup:
    st.subheader("Select a product and z/OS release to see its PTFs / APARs")
    mode = st.radio("Find the product by", ["Pick from lists", "Smart search (typo-tolerant)"], horizontal=True)
    selected: list[int] = []

    if mode == "Pick from lists":
        c1, c2, c3 = st.columns(3)
        vendor = c1.selectbox("Vendor", ["All vendors"] + VENDORS)
        pool = df if vendor == "All vendors" else df[df["vendor"] == vendor]
        product = c2.selectbox("Product", sorted(pool["product"].unique()), index=None,
                               placeholder="Choose a product...")
        if product:
            sub = pool[pool["product"] == product]
            version = c3.selectbox("Version", ["All versions"] + sorted(sub["version"].unique()))
            if version != "All versions":
                sub = sub[sub["version"] == version]
            selected = list(sub.index)
    else:
        q = st.text_input("Type part of a product name, vendor or version",
                          placeholder="e.g. db2 13, strobe, abend-aid cics, netmaster")
        hits = model.search(q, k=10)
        if q and not hits:
            st.warning("No close match found. Try fewer or different words.")
        if hits:
            pick = st.selectbox("Closest matches", [i for i, _ in hits],
                                format_func=lambda i: f"{df.loc[i, 'label']}   ({dict(hits)[i]:.0%} match)")
            selected = [pick]
            same = df[(df["product"] == df.loc[pick, "product"]) & (df.index != pick)]
            if len(same) and st.checkbox(f"Also show the other {len(same)} version(s) of this product"):
                selected += list(same.index)

    if selected:
        for n, i in enumerate(selected):
            render_row(i, expanded=(n == 0 or len(selected) <= 3))
    else:
        st.info("Choose a product above to see its status, PTF / APAR IDs and links.")

# ------------------------------------------------------------------ TAB 2: upgrade planner
with tab_plan:
    st.subheader(f"Build a PTF / APAR worklist for your installed products (z/OS {release})")
    st.session_state.setdefault("plan", [])
    all_labels = list(df["label"])
    c1, c2 = st.columns([3, 1])
    c2.write("")
    qv = c1.selectbox("Quick add: every product with PTF/APAR IDs from vendor", ["-"] + VENDORS)
    if c2.button("➕ Add them") and qv != "-":
        add = df[(df["vendor"] == qv) & (df["n_ids"] > 0)]["label"].tolist()
        st.session_state["plan"] = sorted(set(st.session_state["plan"]) | set(add))
    st.multiselect("Products installed at your site", all_labels, key="plan",
                   placeholder="Start typing a product name...")
    plan = df[df["label"].isin(st.session_state["plan"])]
    if plan.empty:
        st.info("Add one or more products to generate a combined worklist.")
    else:
        summ = plan[["vendor", "product", "version", "status", "n_ids", "link"]].copy()
        summ["status"] = summ["status"].map(lambda s: cdb.status_for({"status": s}, release))
        summ.columns = ["Vendor", "Product", "Version", f"z/OS {release} status", "PTF/APAR IDs", "Verification report"]
        st.dataframe(summ, hide_index=True, **_STRETCH, column_config=LINK_CFG)
        recs = cdb.ptf_records(plan, release)
        st.markdown(f"**{len(recs)} PTF/APAR entries** ({recs['ID'].nunique()} unique IDs)")
        st.dataframe(recs, hide_index=True, **_STRETCH, column_config=LINK_CFG)
        st.download_button("⬇️ Download worklist (CSV)", csv_bytes(recs), file_name=f"worklist_zos{release}.csv")

# ------------------------------------------------------------------ TAB 3: dashboard
with tab_dash:
    st.subheader("z/OS 3.2 compatibility overview")
    st.caption("The sheet assesses z/OS 3.2 only; there is no separate z/OS 3.1 result in the data.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Products / versions", len(fdf))
    c2.metric("🟢 Compatible", int((fdf["status"] == "Compatible").sum()))
    c3.metric("🔴 Not compatible", int((fdf["status"] == "Not compatible").sum()))
    c4.metric("🟡 Unverified / no info", int(fdf["status"].isin(["Unverified", "No info"]).sum()))
    c5.metric("With PTF/APAR IDs", int((fdf["n_ids"] > 0).sum()))
    st.divider()
    chart = fdf.pivot_table(index="vendor", columns="status", values="product", aggfunc="count", fill_value=0)
    chart = chart.reindex(columns=[s for s in cdb.STATUSES if s in chart.columns])
    st.bar_chart(chart)
    st.divider()
    pick_status = st.multiselect("Show statuses", cdb.STATUSES, default=cdb.STATUSES, key="dash_status")
    m = fdf[fdf["status"].isin(pick_status)][["vendor", "product", "version", "status", "id_list", "link"]].copy()
    m.columns = ["Vendor", "Product", "Version", "z/OS 3.2 status", "PTF/APAR IDs", "Verification report"]
    st.dataframe(m, hide_index=True, **_STRETCH, column_config=LINK_CFG)

# ------------------------------------------------------------------ TAB 4: worklist
with tab_ptf:
    st.subheader("PTF / APAR worklist")
    all_rel = st.checkbox("Include IDs for every release mentioned in the sheet", value=False)
    recs = cdb.ptf_records(fdf, None if all_rel else release)
    st.warning(f"{len(recs)} PTF/APAR entries across {recs['Product'].nunique()} products "
               f"({'all releases' if all_rel else 'z/OS ' + release}).")
    f1, f2 = st.columns(2)
    types = f1.multiselect("Type", ["PTF", "APAR"], default=["PTF", "APAR"])
    text = f2.text_input("Filter by ID / product", key="ptf_filter")
    recs = recs[recs["Type"].isin(types)]
    if text:
        recs = recs[recs["ID"].str.contains(text, case=False) | recs["Product"].str.contains(text, case=False)]
    st.dataframe(recs, hide_index=True, **_STRETCH, column_config=LINK_CFG)
    st.download_button("⬇️ Download (CSV)", csv_bytes(recs), file_name="ptf_apar_worklist.csv")
    st.caption("Links are searches / vendor portals; most vendor portals need a customer login. "
               "Cells in the sheet that say 'PTF not found' or 'details not mentioned' have no ID to link.")

# ------------------------------------------------------------------ TAB 5: inventory
with tab_full:
    st.subheader("All compatibility records")
    q = st.text_input("Search product, vendor or version (typo-tolerant)",
                      placeholder="e.g. Strobe, CICS, DB2...", key="inv_q")
    disp = fdf
    if q:
        ids = [i for i, _ in model.search(q, k=50) if i in fdf.index]
        disp = fdf.loc[ids]
    show = disp[["vendor", "product", "version", "status", "status_raw", "id_list", "ptf_text",
                 "comments", "link", "login_free"]].copy()
    show.columns = ["Vendor", "Product", "Version", "z/OS 3.2 status", "Raw answer", "PTF/APAR IDs",
                    "PTF column", "Comments", "Verification report", "Details w/o login"]
    st.dataframe(show, hide_index=True, **_STRETCH, column_config=LINK_CFG)
    st.download_button("⬇️ Download (CSV)", csv_bytes(show), file_name="compatibility_inventory.csv")
