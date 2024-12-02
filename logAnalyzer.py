import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import os
import openai
import re
from dotenv import load_dotenv
# import config  # Import the configuration file
import magic
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from dotenv import load_dotenv, find_dotenv
load = load_dotenv(find_dotenv())
# import config  # Import the configuration file
 
# Load environment variables
# load_dotenv()
 
# Retrieve the API key from the environment variable
api_key = os.getenv("GEMINI_ACCESS_TOKEN")
# Initialize the OpenAI client with the API key
genai.api_key = api_key
 
# Check if the API key is correctly loaded
if api_key is None:
    st.error("API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
 
# Define your ServiceNow credentials (loaded from config.py)
# SERVICENOW_INSTANCE = config.SERVICENOW_INSTANCE
# USERNAME = config.USERNAME
# PASSWORD = config.PASSWORD
SERVICENOW_INSTANCE = os.getenv("VSERVICENOW_INSTANCE")
USERNAME = os.getenv('USERNAMES')
PASSWORD = os.getenv("VPASSWORD")

GEMINI_ENDPOINT = 'https://api.geminiplatform.ai'
GEMINI_ACCESS_TOKEN = os.getenv("GEMINI_ACCESS_TOKEN")

llm = GoogleGenerativeAI(model='gemini-pro',temperature=0, google_api_key= GEMINI_ACCESS_TOKEN)
 
# Initialize magic library
mime = magic.Magic(mime=True)
 

 
# Mapping for state numbers to state names
state_mapping = {
    '1': 'New',
    '2': 'In Progress',
    '3': 'On Hold',
    '4': 'Awaiting Info',
    '5': 'Resolved',
    '6': 'Canceled',
    '7': 'Closed',
    # Add more states if needed
}
 
# Function to fetch incident data from ServiceNow
def fetch_incident_data(incident_number):
    url = f'https://{SERVICENOW_INSTANCE}/api/now/table/incident'
    params = {
        'sysparm_query': f'number={incident_number}',
        'sysparm_limit': 1
    }
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), params=params, verify=False)
    if response.status_code == 200:
        data = response.json()
        if data.get('result'):
            return data['result'][0]
    return None
 
# Function to fetch incident notes from ServiceNow
def fetch_incident_notes(incident_sys_id):
    url = f'https://{SERVICENOW_INSTANCE}/api/now/table/sys_journal_field'
    params = {
        'sysparm_query': f'element_id={incident_sys_id}',
        'sysparm_fields': 'value,sys_created_on,element',
        'sysparm_limit': 100
    }
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), params=params, verify=False)
    if response.status_code == 200:
        data = response.json()
        if data.get('result'):
            return data['result']
    return []
 
# Function to fetch attachments metadata
def fetch_attachments(incident_sys_id):
    url = f'https://{SERVICENOW_INSTANCE}/api/now/table/sys_attachment'
    params = {
        'sysparm_query': f'table_sys_id={incident_sys_id}',
        'sysparm_fields': 'sys_id,file_name'
    }
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), params=params, verify=False)
    if response.status_code == 200:
        data = response.json()
        if data.get('result'):
            return data['result']
    return []
 
# Function to fetch attachment data
def fetch_attachment_data(attachment_sys_id):
    url = f'https://{SERVICENOW_INSTANCE}/api/now/attachment/{attachment_sys_id}/file'
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    if response.status_code == 200:
        return response.content
    return None
 
# Define sensitive information patterns
sensitive_patterns = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{10}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b'
}
 
# Function to redact sensitive information
def redact_sensitive_info(text):
    for pattern in sensitive_patterns.values():
        text = re.sub(pattern, '[REDACTED]', text)
    return text
 
def summarize_text(text, task="Summarize"):
    try:
        prompt = f"{task} the following ticket notes:\n\n{text}\n\nResult:"
        response = llm.invoke(prompt)
        # response = openai.Completion.create(
        #     model="gpt-3.5-turbo-instruct",
        #     prompt=prompt,
        #     max_tokens=100,
        #     n=1,
        #     stop=None,
        #     temperature=0.5
        # )
        result = response#.choices[0].text.strip()
        return result
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
 
# Summarize attachments content
def summarize_attachments(attachments):
    summaries = []
    text_mime_types = ['text/plain', 'text/csv', 'application/json']  # MIME types for text files
 
    for attachment in attachments:
        if 'sys_id' in attachment:
            # Fetch metadata
            url = f'https://{SERVICENOW_INSTANCE}/api/now/table/sys_attachment/{attachment["sys_id"]}'
            response = requests.head(url, auth=HTTPBasicAuth(USERNAME, PASSWORD))
 
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type')
               
                # Check if MIME type indicates a text file
                if any(content_type.startswith(mime_type) for mime_type in text_mime_types):
                    file_data = fetch_attachment_data(attachment['sys_id'])
                    if file_data:
                        file_content = file_data.decode('utf-8', errors='ignore')
                        summary = summarize_text(file_content)
                        summaries.append({
                            "File Name": attachment['file_name'],
                            "Summary": summary
                        })
                else:
                    st.write(f"Skipping non-text file: {attachment['file_name']}")
            else:
                st.write(f"Failed to fetch metadata for attachment: {attachment['file_name']}")
 
    return summaries
 
# Summarizer function for incident notes
def summarize_incident(incident_data, incident_notes):
    combined_notes = ""
    for note in incident_notes:
        if 'element' in note and 'sys_created_on' in note and 'value' in note:
            note_type = 'Work Note' if note['element'] == 'work_notes' else 'Additional Comment'
            note_content = f"{note_type} ({note['sys_created_on']}): {note['value']}\n"
            combined_notes += note_content
        else:
            st.write("Skipping note due to missing fields:", note)
   
    # Redact sensitive information
    cleaned_notes = redact_sensitive_info(combined_notes)
   
    # Summarize the cleaned notes using OpenAI API
    summarized_notes = summarize_text(cleaned_notes, "Summarize")
    incident_summary = {
        "Incident Number": f"{incident_data['number']}",
        "Description": incident_data['description'],
        "Priority": incident_data['priority'],
        "Resolved At": incident_data['resolved_at'],
        "Opened At": incident_data['opened_at'],
        "State": state_mapping.get(incident_data['state'], "Unknown State")
    }
   
    return incident_summary, summarized_notes
 
# Function to fetch detailed resolution steps using GPT-3.5 chat-based model
def fetch_detailed_resolution_steps(incident_data, incident_notes):
    combined_notes = ""
    for note in incident_notes:
        if 'element' in note and 'sys_created_on' in note and 'value' in note:
            note_type = 'Work Note' if note['element'] == 'work_notes' else 'Additional Comment'
            note_content = f"{note_type} ({note['sys_created_on']}): {note['value']}\n"
            combined_notes += note_content
   
    # Redact sensitive information
    cleaned_notes = redact_sensitive_info(combined_notes)
   
    # Generate detailed resolution steps using OpenAI chat-based API
    try:
        prompt = f"You are an IT support specialist tasked with resolving assigned issue. Below is the detailed incident information. Your goal is to provide a step-by-step resolution process that includes all necessary Linux commands, checks, and troubleshooting steps. Ensure that the steps are clear and easy to follow for someone with a basic understanding of IT.{incident_data['number']}:\n\n{cleaned_notes}\n\nPlease generate a detailed resolution process for the above incident."
        response = llm.invoke(prompt)
        # response = genai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",  # Use a model suitable for detailed instructions
        #     messages=[
        #         {"role": "user", "content": prompt}
        #     ],
        #     max_tokens=200,
        #     temperature=0.7
        # )
        detailed_steps = response#['choices'][0]['message']['content'].strip()
        return detailed_steps
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
 
# Streamlit app interface
st.title("ServiceNow Incident Summarizer")
 
# Initialize session state
if 'incident_data' not in st.session_state:
    st.session_state.incident_data = None
    st.session_state.incident_notes = None
    st.session_state.summarized_notes = None
    st.session_state.resolution_steps = None
    st.session_state.incident_number = None  # Add incident number to session state
 
# Text input for incident number
incident_number = st.session_state.incident_number
incident_number = st.text_input("Enter Incident Number:", value=incident_number if incident_number else "")
 
# Layout for buttons in a single line
col1, col2, col3 = st.columns(3)
 
# Button functionalities
with col1:
    if st.button(" Incident Summary "):
        if incident_number:
            st.session_state.incident_number = incident_number
            incident_data = fetch_incident_data(incident_number)
            if incident_data:
                incident_notes = fetch_incident_notes(incident_data['sys_id'])
                incident_summary, summarized_notes = summarize_incident(incident_data, incident_notes)
               
                # Store data in session state
                st.session_state.incident_data = incident_data
                st.session_state.incident_notes = incident_notes
                st.session_state.summarized_notes = summarized_notes
                st.session_state.resolution_steps = None  # Reset resolution steps
               
                # Clear resolution steps if previously set
                if st.session_state.resolution_steps:
                    st.session_state.resolution_steps = None
                    st.experimental_rerun()  # Rerun the app to clear older output
 
            else:
                st.error("Incident not found.")
 
with col2:
    if st.button("Resolution Steps"):
        if incident_number:
            incident_data = fetch_incident_data(incident_number)
            if incident_data:
                # Check if incident state is Closed or Resolved
                if incident_data['state'] in ['6', '5', '7']:  # Assuming state '6' is Closed and '5' is Resolved
                    st.info("Resolution steps are not available for Canceled or Closed or Resolved incidents.")
                else:
                    incident_notes = fetch_incident_notes(incident_data['sys_id'])
                   
                    # Fetch detailed resolution steps
                    resolution_steps = fetch_detailed_resolution_steps(incident_data, incident_notes)
                   
                    # Store data in session state
                    st.session_state.incident_data = incident_data
                    st.session_state.incident_notes = incident_notes
                    st.session_state.resolution_steps = resolution_steps
                   
                    # Clear summarized notes if previously set
                    if st.session_state.summarized_notes:
                        st.session_state.summarized_notes = None
                        st.experimental_rerun()  # Rerun the app to clear older output
 
            else:
                st.error("Incident not found.")
        else:
            st.error("Please enter an incident number.")
 
with col3:
    if st.button("Clear"):
        st.session_state.incident_data = None
        st.session_state.incident_notes = None
        st.session_state.summarized_notes = None
        st.session_state.resolution_steps = None
        st.session_state.incident_number = None  # Reset incident number
        st.rerun()
 
# Display incident summary and notes
if st.session_state.incident_data:
    incident_summary_df = pd.DataFrame([{
        "Incident Number": st.session_state.incident_data['number'],
        "Description": st.session_state.incident_data['description'],
        "Priority": st.session_state.incident_data['priority'],
        "Resolved At": st.session_state.incident_data['resolved_at'],
        "Opened At": st.session_state.incident_data['opened_at'],
        "State": state_mapping.get(st.session_state.incident_data['state'], "Unknown State")
    }])
    st.write(incident_summary_df.set_index('Incident Number'))
   
    if st.session_state.summarized_notes:
        st.text_area("Summarized Incident Notes", value=st.session_state.summarized_notes, height=200)
   
    if st.session_state.resolution_steps:
        st.text_area("Resolution Steps", value=st.session_state.resolution_steps, height=200)
 
 
# Fetch and display attachments
if st.session_state.incident_data:
    attachments = fetch_attachments(st.session_state.incident_data['sys_id'])
    if attachments:
        st.write("Attachments:")
        attachment_summaries = summarize_attachments(attachments)
        for attachment_summary in attachment_summaries:
            st.write(f"Filename: {attachment_summary['File Name']}")
            st.write(f"Summary: {attachment_summary['Summary']}")
            if 'sys_id' in attachment_summary:
                file_data = fetch_attachment_data(attachment_summary['sys_id'])
                if file_data:
                    st.download_button(label=f"Download {attachment_summary['File Name']}", data=file_data, file_name=attachment_summary['File Name'])
    else:
        st.write("No attachments found.")
 
 
