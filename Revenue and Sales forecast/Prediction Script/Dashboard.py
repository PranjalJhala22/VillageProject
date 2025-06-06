import streamlit as st
import pandas as pd
import altair as alt

# 1. Load CSV with caching
@st.cache  # use @st.cache_data if you’ve upgraded Streamlit
def load_data(path: str):
    df = pd.read_csv(path, parse_dates=["Session Date"])
    return df

df = load_data("predicted_item_sales.csv")

# 2. Sidebar filters
st.sidebar.header("Filters")

# 2a. Date range
min_date = df["Session Date"].min().date()
max_date = df["Session Date"].max().date()
start_date, end_date = st.sidebar.date_input(
    "Session Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# 2b. Session hour
all_hours = sorted(df["Session Hour"].unique())
selected_hours = st.sidebar.multiselect(
    "Session Hour",
    options=all_hours,
    default=all_hours
)

# 2c. Item classes (hard-coded)
item_classes = [
    "SNACK - CHIPS",
    "FOOD - VJUNIOR",
    "ICE CREAMS - OTHER",
    "DRINKS - EXTRA LARGE",
    "DRINKS",
    "DRINKS - SMALL",
    "DRINKS - MEDIUM",
    "ICE CREAMS - CHOC TO",
    "DRINKS - NO ICE",
    "DRINKS - LARGE",
    "POPCORN",
]
selected_items = st.sidebar.multiselect(
    "Item Classes",
    options=item_classes,
    default=item_classes
)

# 3. Apply filters
filtered_df = df[
    (df["Session Date"].dt.date >= start_date) &
    (df["Session Date"].dt.date <= end_date) &
    (df["Session Hour"].isin(selected_hours))
]

st.write(f"Showing {len(filtered_df)} sessions between {start_date} and {end_date} for hours {selected_hours}.")
st.dataframe(filtered_df.head())

# 4. KPI cards
total_revenue = filtered_df["Total Session Revenue"].sum()
total_admits  = filtered_df["Total Admits"].sum()
total_items   = filtered_df[selected_items].sum().sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Expected Revenue", f"${total_revenue:,.2f}")
# col2.metric("Total People throughout hours(Exploded)", int(total_admits))
col3.metric("Total Expected Items Sold", int(total_items))

# 5a. Overall item breakdown
st.subheader("Overall Item Breakdown")
overall_counts = (
    filtered_df[selected_items]
    .sum()
    .rename_axis("item_class")
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
overall_chart = (
    alt.Chart(overall_counts)
    .mark_bar()
    .encode(
        x=alt.X("item_class:N", sort="-y", title="Item Class"),
        y=alt.Y("count:Q", title="Total Predicted Count"),
        tooltip=["item_class","count"]
    )
    .properties(width="container", height=300)
)
st.altair_chart(overall_chart, use_container_width=True)

# 5b. Session-level breakdown
st.subheader("Session-level Item Breakdown")
date_options = sorted(filtered_df["Session Date"].dt.date.unique())
sel_date = st.selectbox("Choose Session Date", date_options)
hour_options = sorted(
    filtered_df[filtered_df["Session Date"].dt.date == sel_date]["Session Hour"].unique()
)
sel_hour = st.selectbox("Choose Session Hour", hour_options)

session_slice = filtered_df[
    (filtered_df["Session Date"].dt.date == sel_date) &
    (filtered_df["Session Hour"] == sel_hour)
]
session_counts = (
    session_slice[selected_items]
    .sum()
    .rename_axis("item_class")
    .reset_index(name="count")
)
session_chart = (
    alt.Chart(session_counts)
    .mark_bar()
    .encode(
        x=alt.X("item_class:N", sort="-y", title="Item Class"),
        y=alt.Y("count:Q", title="Count"),
        tooltip=["item_class","count"]
    )
    .properties(width="container", height=300)
)
st.altair_chart(session_chart, use_container_width=True)

# 5c. Stacked composition over time
st.subheader("Item Composition Over Time")
group_by = st.radio("Group by:", ["Date", "Hour"], horizontal=True)

if group_by == "Date":
    df_time = filtered_df.copy()
    df_time["Session Day"] = df_time["Session Date"].dt.date
    group_df = (
        df_time.groupby("Session Day")[selected_items]
        .sum()
        .reset_index()
        .melt(id_vars=["Session Day"], var_name="item_class", value_name="count")
    )
    x_enc = alt.X("Session Day:T", title="Date")
else:
    group_df = (
        filtered_df.groupby("Session Hour")[selected_items]
        .sum()
        .reset_index()
        .melt(id_vars=["Session Hour"], var_name="item_class", value_name="count")
    )
    x_enc = alt.X("Session Hour:O", title="Hour of Day")

stacked_chart = (
    alt.Chart(group_df)
    .mark_area(interpolate="step")
    .encode(
        x=x_enc,
        y=alt.Y("count:Q", stack="normalize", title="Proportion"),
        color=alt.Color("item_class:N", legend=alt.Legend(title="Item Class")),
        tooltip=["item_class","count"]
    )
    .properties(width="container", height=350)
)
st.altair_chart(stacked_chart, use_container_width=True)




# --- Donut charts for Language, Genre & Rating (and Duration if you like) ---
st.subheader("Attribute Distributions (Filtered Sessions)")

attr_groups = {
    "Language": [
        "Lang_Assamese","Lang_Bengali","Lang_Chinese (Cantonese)",
        "Lang_Chinese (Mandarin)","Lang_English","Lang_Filipino",
        "Lang_Gujarati","Lang_Hindi","Lang_Indonesian","Lang_Japanese",
        "Lang_Kannada","Lang_Korean","Lang_Malayalam","Lang_Maori",
        "Lang_Nepali","Lang_No Subtitles","Lang_Not assigned",
        "Lang_Punjabi","Lang_Tamil","Lang_Telugu","Lang_Thai",
        "Lang_Urdu","Lang_Vietnamese"
    ],
    "Genre": [
        "Genre_ACTION","Genre_ADVENTURE","Genre_ANIMATION","Genre_BIOGRAPHY",
        "Genre_COMEDY","Genre_CRIME","Genre_DOCUMENTARY","Genre_DRAMA",
        "Genre_FAMILY","Genre_FANTASY","Genre_GAMING","Genre_HORROR",
        "Genre_MUSIC","Genre_MUSICAL","Genre_MYSTERY","Genre_ROMANCE",
        "Genre_SCI-FI","Genre_THRILLER","Genre_TO BE ADVISED"
    ],
    "Rating": [
        "Rating_CTC","Rating_E","Rating_G","Rating_M","Rating_MA15",
        "Rating_PG","Rating_R18+"
    ]
    # optionally add "Duration": [...]
}

if filtered_df.empty:
    st.write("No sessions in this slice.")
else:
    for title, cols in attr_groups.items():
        # 1) Sum each one‐hot column across **all** filtered sessions
        df_agg = (
            filtered_df[cols]
            .sum()
            .reset_index()
            .rename(columns={"index": title, 0: "count"})
        )
        # 2) Drop zero‐count attributes
        df_agg = df_agg[df_agg["count"] > 0]

        # 3) Build a donut chart
        chart = (
            alt.Chart(df_agg)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("count:Q", title=None),
                color=alt.Color(f"{title}:N", legend=alt.Legend(title=title)),
                tooltip=[f"{title}:N", "count:Q"]
            )
            .properties(width=300, height=300, title=title)
        )
        st.altair_chart(chart, use_container_width=False)








############vista breakdown

# --- Item-Level Stock Breakdown ---
st.subheader("Recommended Item Breakdown for Stocking(Based on previous trends)")

# Load percentage breakdown Excel (if not already loaded at top)
@st.cache
def load_breakdown(path: str):
    return pd.read_excel(path)

vista_breakdown = load_breakdown("Vista_Item_Percentage_Breakdown.xlsx")

# Compute total predicted items per class from filtered data
item_totals = (
    filtered_df[selected_items]
    .sum()
    .reset_index()
    .rename(columns={"index": "Item Class", 0: "Total Predicted Items"})
)

# Merge totals with breakdown percentages
merged = pd.merge(vista_breakdown, item_totals, on="Item Class", how="inner")
merged["Recommended Quantity"] = (merged["Percentage"] / 100) * merged["Total Predicted Items"]
merged["Recommended Quantity"] = merged["Recommended Quantity"].round()

# Show interactive selector
sel_class = st.selectbox("Choose Item Class for Breakdown", sorted(merged["Item Class"].unique()))
class_df = merged[merged["Item Class"] == sel_class][["VISTA Item", "Recommended Quantity"]].sort_values(by="Recommended Quantity", ascending=False)

# Display table
st.write(f"**Breakdown for {sel_class}**")
st.dataframe(class_df)


# # --- Donut Chart for Selected Item Class Breakdown ---
# st.subheader(f"Visual Breakdown for {sel_class}")

# if not class_df.empty:
#     donut_chart = (
#         alt.Chart(class_df)
#         .mark_arc(innerRadius=50)
#         .encode(
#             theta=alt.Theta("Recommended Quantity:Q", title=None),
#             color=alt.Color("VISTA Item:N", legend=alt.Legend(title="VISTA Item")),
#             tooltip=["VISTA Item", "Recommended Quantity"]
#         )
#         .properties(width=400, height=400)
#     )
#     st.altair_chart(donut_chart, use_container_width=False)
# else:
#     st.write("No data available for donut chart.")


# # --- Donut Chart for Percentage Breakdown ---
# st.subheader(f"Percentage Breakdown for {sel_class}")

# # Merge class_df with percentage info from the original breakdown for full context
# percent_df = vista_breakdown[vista_breakdown["Item Class"] == sel_class][["VISTA Item", "Percentage"]]

# if not percent_df.empty:
#     percent_donut = (
#         alt.Chart(percent_df)
#         .mark_arc(innerRadius=50)
#         .encode(
#             theta=alt.Theta("Percentage:Q", title=None),
#             color=alt.Color("VISTA Item:N", legend=alt.Legend(title="VISTA Item")),
#             tooltip=["VISTA Item", alt.Tooltip("Percentage:Q", format=".2f")]
#         )
#         .properties(width=400, height=400)
#     )
#     st.altair_chart(percent_donut, use_container_width=False)
# else:
#     st.write("No percentage data available for this item class.")




# --- Side-by-side Donut Charts for Quantity & Percentage ---
st.subheader(f"Breakdown for {sel_class}")

# Prepare both dataframes
percent_df = vista_breakdown[vista_breakdown["Item Class"] == sel_class][["VISTA Item", "Percentage"]]

# Only build if data is available
if not class_df.empty and not percent_df.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Recommended Quantity**")
        donut_qty = (
            alt.Chart(class_df)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("Recommended Quantity:Q", title=None),
                color=alt.Color("VISTA Item:N", legend=None),
                tooltip=["VISTA Item", "Recommended Quantity"]
            )
            .properties(width=350, height=350)
        )
        st.altair_chart(donut_qty, use_container_width=False)

    with col2:
        st.markdown("**Percentage Breakdown**")
        donut_pct = (
            alt.Chart(percent_df)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("Percentage:Q", title=None),
                color=alt.Color("VISTA Item:N", legend=alt.Legend(title="VISTA Item")),
                tooltip=["VISTA Item", alt.Tooltip("Percentage:Q", format=".2f")]
            )
            .properties(width=350, height=350)
        )
        st.altair_chart(donut_pct, use_container_width=False)
else:
    st.write("Not enough data to display donut charts.")
