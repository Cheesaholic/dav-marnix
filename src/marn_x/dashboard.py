import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from marn_x.full_stopping_and_thinking import (PeriodLoader, PeriodPlotter,
                                               PeriodSettings)
# Import the required classes from existing module
from marn_x.utils.settings import AllVars


def streamlit_period_analysis():
    """
    Streamlit dashboard that performs period analysis:
    1. Load settings
    2. Process the data
    3. Create interactive controls for filtering
    4. Display the visualization
    """
    # Set up the Streamlit page
    st.set_page_config(page_title="Period Usage Analysis Dashboard", layout="wide")
    st.title("📝 Message Ending Punctuation Analysis")

    # Load default settings
    settings = PeriodSettings(**AllVars(file_stem="full_stopping_and_thinking"))

    # Create sidebar for controls
    with st.sidebar:
        st.header("Analysis Settings")

        # Add slider for minimum long messages filter
        min_messages = st.slider(
            "Minimum long messages per author",
            min_value=1,
            max_value=100,
            value=settings.min_long_messages,
            step=1,
            help="Authors with fewer messages than this will be excluded",
        )

        # Update the settings with the slider value
        settings.min_long_messages = min_messages

        # Optional: Add more controls for other parameters
        st.subheader("End of Sentence Markers")
        if st.checkbox("Show/Edit End of Sentence Markers", False):
            markers_text = st.text_input(
                "End of sentence markers",
                value=", ".join(settings.end_of_sentence_list),
            )
            settings.end_of_sentence_list = [m.strip() for m in markers_text.split(",")]

        st.subheader("About")
        st.info(
            "This dashboard analyzes how frequently messages end with "
            "sentence-ending punctuation like periods, question marks, etc."
        )

    # Main content area
    st.subheader("Message Ending Analysis")

    # Show loading message while processing data
    with st.spinner("Loading and processing data..."):
        # Load and process the data
        try:
            loader = PeriodLoader(settings)
            plotter = PeriodPlotter(settings)

            # Display information about the data
            st.write(f"Loaded {len(loader.datafiles.processed)} data points")

            # Apply slider filter
            filtered_data = loader.datafiles.processed[
                loader.datafiles.processed["cnt_long_messages"] >= min_messages
            ]

            # Show how many points remain after filtering
            st.write(f"Showing {len(filtered_data)} data points after filtering")

            # Show filters being applied
            st.info(
                f"⚙️ Applied filters: minimum {min_messages} long messages per author"
            )

            # Display the data table (expandable)
            with st.expander("View Raw Data", expanded=False):
                st.dataframe(filtered_data)

            # Generate the plot
            if len(filtered_data) > 0:
                fig = plotter.plot(filtered_data)
                st.pyplot(fig)
            else:
                st.warning(
                    "No data points match the current filter settings. Try reducing the minimum message count."
                )

        except Exception as e:
            st.error(f"Error loading or processing data: {str(e)}")
            st.error("Please check your data files and settings.")


if __name__ == "__main__":
    streamlit_period_analysis()
