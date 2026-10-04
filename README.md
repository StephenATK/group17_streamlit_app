# Group 17 - Llama customer-service chatbot (no RAG)

## Run order
1. Open Group17_Llama_Chatbot_Colab.ipynb in Google Colab, set a T4 GPU, add the HF_TOKEN secret
   (and optionally ANTHROPIC_API_KEY for judge scores), accept the Llama licence on Hugging Face, run all cells.
2. Download group17_artifacts.zip and unzip it into artifacts/ in this folder.
3. Local check:  pip install -r requirements.txt  then  streamlit run app.py
4. Push this folder to a public GitHub repo, then deploy on Streamlit Community Cloud (main file: app.py).

## Notes
- The app contains no hard-coded results; everything is read from artifacts/.
- Community Cloud cannot host Llama, so the demo page replays real held-out answers. For a live demo, host the
  adapter elsewhere and add HF_ENDPOINT_URL and HF_TOKEN under App settings > Secrets.
