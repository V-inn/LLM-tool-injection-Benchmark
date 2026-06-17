"""
custom_prompts.py — "Custom Prompts" tab.

Lets the user hand-author additional defense system prompts, persisted to
custom_prompts.json (loaded by core.prompts.load_all_prompts on the next run).
"""
import os
import json

import streamlit as st


def render(ctx):
    st.header("Inject Custom System Prompts")
    st.write("Add new defense strategies here. They will be saved to `custom_prompts.json` and loaded by the orchestrator.")

    with st.form("prompt_form"):
        prompt_key = st.text_input("Prompt ID (e.g., S7_CUSTOM_SHIELD)")
        prompt_text = st.text_area("System Prompt Text", height=150)
        submit_prompt = st.form_submit_button("Save Prompt")

        if submit_prompt and prompt_key and prompt_text:
            custom_data = {}
            if os.path.exists(ctx.custom_prompts_file):
                with open(ctx.custom_prompts_file, "r", encoding="utf-8") as f:
                    custom_data = json.load(f)

            custom_data[prompt_key] = prompt_text

            with open(ctx.custom_prompts_file, "w", encoding="utf-8") as f:
                json.dump(custom_data, f, indent=4)
            st.success(f"Prompt '{prompt_key}' saved successfully!")
