import streamlit as st
import streamlit.components.v1 as components

def main():
    st.title("Website Popup Viewer")
    
    # URL input field
    default_url = "https://doj.gov.in/live-streaming-of-court-cases/"
    popup_url = st.text_input("Enter URL to view", value=default_url)
    
    # Checkbox to toggle popup visibility
    show_popup = st.checkbox("Open Popup")
    
    # If popup is checked, display the iframe
    if show_popup:
        st.subheader("Website Popup")
        components.iframe(popup_url, height=600, scrolling=True)

if __name__ == "__main__":
    main()