import os
import json
import sys
import streamlit as st
import feedparser
import vertexai
from langchain.llms import VertexAI
from datetime import datetime
from datetime import timedelta
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate



st.title("RSS Feed Summary Generator")

url = st.text_input("Enter the RSS feed url","https://www.bleepingcomputer.com/feed/")

clicked = st.button("Generate Summary")

#date_range = st.sidebar.date_input(
#    "Select a date range",
#    value=(datetime.now() - timedelta(weeks=4), datetime.now())
#)


secret = st.secrets["gcp_credentials"]

# Define the file path (this file will be created in the same directory as your Streamlit script)
file_path = "service_account.json"

# Write the dictionary to this file
with open(file_path, 'w') as f:
    json.dump(dict(secret), f)

# Set the environment variable to the path of the created file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file_path



vertexai.init(project='sada-prakash')
llm = VertexAI(
    model_name="text-bison@001",
    temperature=0.2, max_output_tokens=1024, top_k=40, top_p=0.8,
    verbose=True,
)

def get_posts(rss_feed_url):
    rss_feeds = feedparser.parse(rss_feed_url)
    posts = []
    for feed in rss_feeds["entries"]:
        post = {
            "title": feed["title"],
            "link": feed["link"],
            "published_ts": feed["published"],
            #"author": feed["author"]
        }
        posts.append(post)

    try:
        posts.sort(key=lambda x: datetime.strptime(x["published_ts"], "%a, %d %b %Y %H:%M:%S %z"), reverse=True)
    except:
        posts.sort(key=lambda x: datetime.strptime(x["published_ts"], "%a, %d %b %Y %H:%M:%S %Z"), reverse=True)

    summaries=[]
    for post in posts[:10]:
        post["summary"] = generate_summary(WebBaseLoader(post["link"]).load())
        summaries.append(post)
    return summaries

def generate_summary(article):
    prompt_template="""
Provide a very short summary, no more than four sentences, for the following article:

{text}

Summary:

"""
    prompt = PromptTemplate(
        input_variables=["text"],
        template=prompt_template,
    )
    chain = load_summarize_chain(llm,chain_type="stuff", prompt=prompt)
    summarized_text = chain.run(article)
    return summarized_text

if clicked:
    if url:
        with st.spinner("Generating summaries for the recent 10 posts..."):
            posts = get_posts(url)
        if posts:
            for post in posts:
                with st.expander(post["title"], expanded=True):
                    st.write(f"summary:\n {post['summary']}")
                    st.write(f"published:\n {post['published_ts']}")
                    #st.write(f"author:\n {post['author']}")
                    st.markdown(f"[link..]({post['link']})")
    else:
        st.write("Enter a rss feed url")
        sys.exit(1)
