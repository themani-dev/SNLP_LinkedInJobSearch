
import streamlit as st
from FinalProject_JobSearch import JobSearch
import json

@st.cache_resource
def load_job_search():
    return JobSearch(file='/Users/mani/PycharmProjects/rootdir/SNLP/hw/out/FinalProject/')

# Create an instance of the JobSearch class
obj = load_job_search()

st.title('LinkedIn Job Search')
query = st.text_input('Enter your query:')
selected_option = st.slider('Select No of results:', min_value=5, max_value=50, value=5, step=1)
if st.button('Search'):
    if query:
        results = obj.ModelSearch(query=query, noresults=int(selected_option))
        for idx, json_obj in enumerate(results):
            title = json_obj['title'] + '\t' + json_obj['work_type'] + '\t' + json_obj['location']
            with st.expander(title):
                st.write(f"Job Id : {json_obj['job_id']}")
                st.write(f"Description : \n\n{json_obj['description']}")
                if json_obj['remote_allowed'] is None:
                    r_flag = 'NO'
                elif int(json_obj['remote_allowed']) == 1:
                    r_flag = 'Yes'
                else:
                    r_flag = 'NO'
                st.write(f"Remote Allowed : {r_flag}")
                if json_obj['sponsored'] is None:
                    s_flag = 'NO'
                elif int(json_obj['sponsored']) == 1:
                    s_flag = 'Yes'
                else:
                    s_flag = 'NO'
                st.write(f"Sponsorship Available : {s_flag}")
                # for key, value in json_obj.items():
                #     st.write(f"{key}: {value}")  # Display key-value pairs
                st.markdown(f"[Click here for more information]({json_obj['job_posting_url']})")
    else:
        st.warning("Please enter a query.")
