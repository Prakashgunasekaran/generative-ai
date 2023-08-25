import os
import json
import sys
import streamlit as st
import feedparser
import vertexai
from langchain.llms import VertexAI
from datetime import datetime
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate


st.title("RSS News Feed Summary Generator")

url = st.sidebar.text_input("Enter the RSS feed url", "http://feeds.bbci.co.uk/news/rss.xml")

creative_level = st.sidebar.radio("Choose creative level", ("Low", "Medium", "High"), index=0)
clicked = st.sidebar.button("Generate Summary")

# Retrieve gcp service account key information from streamlit secret
secret = st.secrets["gcp_credentials"]

service_account_key_file = "service_account.json"

# Write the service account key to the file
with open(service_account_key_file, 'w') as f:
    json.dump(dict(secret), f)

# Set the environment variable to the path of the created file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_file

model_params = {
    "Low": {
        "temperature": 0.1,
        "top_p": 0.1,
        "top_k": 1
    },
    "Medium": {
        "temperature": 0.5,
        "top_p": 0.5,
        "top_k": 20
    },
    "High": {
        "temperature": 1,
        "top_p": 1,
        "top_k": 40
    }
}
# initialize vertex ai session
vertexai.init(project='sada-prakash')


def get_summaries(rss_feed_url, creative_level):
    """function to read the rss feed and generate summaries"""
    rss_feeds = feedparser.parse(rss_feed_url)
    temp_posts = []
    for feed in rss_feeds["entries"]:
        if "published" in feed:
            temp_post = {
                "title": feed["title"],
                "link": feed["link"],
                "published_ts": feed["published"]
            }
            temp_posts.append(temp_post)
        elif "pubDate" in feed:
            temp_post = {
                "title": feed["title"],
                "link": feed["link"],
                "published_ts": feed["pubDate"]
            }
            temp_posts.append(temp_post)


    # match the published time format from the feed and sort the feeds based on published time in descending order
    try:
        temp_posts.sort(key=lambda x: datetime.strptime(x["published_ts"], "%a, %d %b %Y %H:%M:%S %z"), reverse=True)
    except:
        temp_posts.sort(key=lambda x: datetime.strptime(x["published_ts"], "%a, %d %b %Y %H:%M:%S %Z"), reverse=True)

    summaries = []

    llm = VertexAI(
        model_name="text-bison@001",
        temperature=creative_level["temperature"],
        max_output_tokens=256,
        top_k=creative_level["top_k"],
        top_p=creative_level["top_p"]
    )
    for temp_post in temp_posts[:10]:  # invoke generate_summary function for the recent 10 posts from the list
        output = generate_summary(WebBaseLoader(temp_post["link"]).load(), llm)
        if output:
            summary, topic = output
            if summary:
                temp_post["summary"] = summary
                temp_post["topic"] = topic
                summaries.append(temp_post)
    return summaries


def generate_summary(article, llm):
    """function to generate summary of an article using langchain"""
    summary_prompt_template = """
Provide a very short summary, no more than four sentences, for the following article:

{text}

Summary:

"""
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template=summary_prompt_template,
    )

    summary_chain = load_summarize_chain(llm, chain_type="stuff", prompt=summary_prompt, output_key="summary")

    topic_prompt_template="""Classify the following text into one of several predefined topics, such as General News, Sports, Politics, Media, Entertainment, Business, Technology, Health . Any text that cannot be clasified falls into 'General News' class\n\n
    
 text: If you can't afford your energy bills, you should check your direct debit, pay what you can, claim what you are entitled to and adjust your boiler.
 class: Business
 
 text: Nottinghamshire Police officer hit by train while helping man
 class: General News
 
 text:The US has sensitive nuclear technology at a nuclear power plant inside Ukraine and is warning Russia not to touch it, according to a letter the US Department of Energy sent to Russia’s state-owned nuclear energy firm Rosatom last month
 class: Politics
 
 text: Fox News settled with Dominion Voting Systems on Tuesday for $787.5 million, just hours before the case was set to go to trial
 class: Media
 
 text: {summary}
 class: 
"""
    topic_prompt = PromptTemplate(
        input_variables=["summary"],
        template=topic_prompt_template,
    )
    topic_chain = LLMChain(llm=llm, prompt=topic_prompt, output_key="class")

    overall_chain = SequentialChain(
        chains=[summary_chain, topic_chain],
        input_variables=["input_documents"],
        output_variables=["summary", "class"],
        verbose=True
    )
    try:
        output = overall_chain(article)
        return output["summary"], output["class"]
    except:
        pass


# when the Generate Summary button is submitted
if clicked:
    if url:
        with st.spinner("Generating summaries for the recent 10 posts..."):
            # invoke the function to generate summaries
            posts = get_summaries(url, model_params[creative_level])
        if posts:
            for post in posts:
                with st.expander(post["title"], expanded=True):
                    st.markdown(f"**Topic:** {post['topic']}")
                    st.markdown(f"**Summary:** {post['summary']}")
                    st.markdown(f"**Published:** {post['published_ts']}")
                    st.markdown(f"[link..]({post['link']})")
    else:
        st.write("Enter a rss feed url")
        sys.exit(1)
