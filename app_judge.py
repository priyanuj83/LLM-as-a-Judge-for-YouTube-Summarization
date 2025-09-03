import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from summarizer_judge import load_summarizers, generate_summaries
from judge import judge_summary

# Load models once
summarizers = load_summarizers()

st.title("🎥 YouTube Abstractive Summarizer with LLM-as-a-Judge")

# User input
video_url = st.text_input("Enter YouTube Video URL")

if video_url:
    try:
        video_id = video_url.split("v=")[-1]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([t["text"] for t in transcript_list])

        st.subheader("Transcript Extracted ✅")
        with st.expander("Show Transcript"):
            st.write(transcript)

        # Generate candidate summaries
        st.subheader("Candidate Summaries")
        summaries = generate_summaries(summarizers, transcript)
        for name, summary in summaries.items():
            st.markdown(f"**{name.upper()} Summary:** {summary}")

        # Judge the best summary
        st.subheader("Judge’s Decision 🏆")
        best_summary = judge_summary(transcript, summaries)
        st.write(best_summary)

    except Exception as e:
        st.error(f"Error: {e}")