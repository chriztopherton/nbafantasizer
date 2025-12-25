"""AI Analysis tab functionality for Fantasy Points Analyzer."""

import sys

import streamlit as st
from ai_model import LANGCHAIN_AVAILABLE, LANGCHAIN_ERROR, PlayerAnalysisChatbot


def render_ai_analysis_tab(player_name, player_data_ytd, aggregated_df, injury_scraper):
    """
    Render the AI Analysis sub-tab within Player Analysis.

    Args:
        player_name (str): Name of the selected player.
        player_data_ytd (pd.DataFrame): Filtered player data for the date range.
        aggregated_df (pd.DataFrame): Aggregated statistics by time window.
        injury_scraper: Injury scraper instance for fetching injury information.
    """
    st.subheader("🤖 AI Player Performance Analysis")
    st.markdown(
        "Get AI-powered insights about this player's performance and fantasy value for Yahoo Fantasy Basketball managers."
    )

    # Check if AI is available
    if not LANGCHAIN_AVAILABLE:
        python_path = sys.executable
        venv_detected = "venv" in python_path or ".venv" in python_path

        st.warning(
            "⚠️ AI analysis requires additional packages. Install with: `pip install langchain-openai langchain`"
        )

        if LANGCHAIN_ERROR:
            with st.expander("🔍 Debug Information"):
                st.code(f"Import Error: {LANGCHAIN_ERROR}", language="text")
                st.code(f"Python Path: {python_path}", language="text")
                st.code(f"Virtual Env Detected: {venv_detected}", language="text")

        if not venv_detected:
            st.error(
                "🔴 **Virtual Environment Not Active!**\n\n"
                "It looks like you have a `venv` directory with the packages installed, but Streamlit is running "
                "with a different Python interpreter. Please:\n\n"
                "1. Activate your virtual environment:\n"
                "   ```bash\n"
                "   source venv/bin/activate  # On macOS/Linux\n"
                "   # or\n"
                "   venv\\Scripts\\activate  # On Windows\n"
                "   ```\n\n"
                "2. Then run Streamlit again:\n"
                "   ```bash\n"
                "   streamlit run src/FP_analyzer/app.py\n"
                "   ```"
            )
        else:
            st.info(
                "Once installed, set your `OPENAI_API_KEY` in your `.env` file to enable AI analysis."
            )
    else:
        # Initialize chatbot in session state
        # Check if chatbot exists and is valid (has new structure without 'chain')
        needs_reinit = False
        if "player_chatbot" not in st.session_state:
            needs_reinit = True
        elif hasattr(st.session_state.player_chatbot, "chain"):
            # Old version with 'chain' attribute - needs reinitialization
            needs_reinit = True
            st.info("🔄 Updating AI chatbot to new version...")
        elif not hasattr(st.session_state.player_chatbot, "llm"):
            # Missing required attributes - needs reinitialization
            needs_reinit = True

        if needs_reinit:
            try:
                st.session_state.player_chatbot = PlayerAnalysisChatbot()
                if "ai_chat_messages" not in st.session_state:
                    st.session_state.ai_chat_messages = []
            except ValueError as e:
                st.error(f"❌ Configuration Error: {str(e)}")
                st.info(
                    "Please set your `OPENAI_API_KEY` in your `.env` file to enable AI analysis."
                )
                st.stop()
            except Exception as e:
                st.error(f"❌ Error initializing AI chatbot: {str(e)}")
                st.stop()

        # Get injury info for the summary
        try:
            injury_info = injury_scraper.get_player_injury(player_name)
        except Exception:
            injury_info = None

        # Generate summary button
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(
                "📊 Generate Player Performance Summary",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("🤖 Analyzing player performance..."):
                    summary = st.session_state.player_chatbot.generate_player_summary(
                        player_name=player_name,
                        player_data=player_data_ytd,
                        aggregated_stats=aggregated_df,
                        injury_info=injury_info,
                    )
                    # Store summary in session state
                    st.session_state.player_summary = summary
                    st.session_state.summary_generated = True

        with col2:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.ai_chat_messages = []
                st.session_state.player_chatbot.clear_memory()
                if "player_summary" in st.session_state:
                    del st.session_state.player_summary
                if "summary_generated" in st.session_state:
                    del st.session_state.summary_generated
                st.rerun()

        # Display summary if generated
        if "summary_generated" in st.session_state and st.session_state.summary_generated:
            if "player_summary" in st.session_state:
                st.markdown("---")
                st.markdown("### 📝 Player Performance Summary")
                st.markdown(st.session_state.player_summary)

        # Chat interface for follow-up questions
        st.markdown("---")
        st.markdown("### 💬 Ask Follow-up Questions")

        # Display chat messages
        for message in st.session_state.ai_chat_messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])

        # Chat input
        if prompt := st.chat_input(
            f"Ask about {player_name}'s fantasy value, trends, or trade advice..."
        ):
            # Add user message to history
            st.session_state.ai_chat_messages.append({"role": "user", "content": prompt})

            # Show typing indicator
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Add context about the current player
                    contextual_prompt = f"Context: We're analyzing {player_name}. {prompt}"
                    bot_response = st.session_state.player_chatbot.get_response(contextual_prompt)
                    st.session_state.ai_chat_messages.append(
                        {"role": "assistant", "content": bot_response}
                    )

            # Rerun to display new messages
            st.rerun()

        # Helpful suggestions
        st.markdown("---")
        st.markdown("**💡 Try asking:**")
        st.markdown(f"- What's {player_name}'s consistency like?")
        st.markdown(f"- Should I buy, sell, or hold {player_name}?")
        st.markdown(f"- What's {player_name}'s fantasy ceiling and floor?")
        st.markdown(f"- How does {player_name} compare to other players at their position?")
        st.markdown(f"- What are the concerns about {player_name}'s performance?")
