
import streamlit as st
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load data
df = pd.read_csv("customer_reviews.csv")

st.title("Banking Customer Feedback Dashboard")

st.subheader("1. Sample Customer Reviews")
st.dataframe(df.head(10))

st.subheader("2. Sentiment Distribution")
sentiment_counts = df['sentiment'].value_counts()
st.bar_chart(sentiment_counts)

st.subheader("3. Word Cloud of All Feedback")
all_text = " ".join(df["review"])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

st.subheader("4. Topic Modeling (LDA)")
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["review"])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)
terms = vectorizer.get_feature_names_out()

topics = []
for i, topic in enumerate(lda.components_):
    top_terms = [terms[i] for i in topic.argsort()[:-6:-1]]
    topics.append(f"Topic {i+1}: " + ", ".join(top_terms))

for topic in topics:
    st.markdown(f"- {topic}")

st.subheader("5. Simulated GenAI Insight Response")
question = st.text_input("Ask a question about customer feedback")
if question:
    st.markdown("**Simulated Answer:**")
    st.markdown("Customers are primarily frustrated with mobile app reliability and branch availability. "
                "Sentiment trends indicate dissatisfaction peaking during weekends. "
                "Recommendation: Improve mobile app stability and reduce in-branch wait times.")
