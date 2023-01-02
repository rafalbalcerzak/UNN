import streamlit as st
import io
import pandas as pd


# To run install streamlit: pip install streamlit
# RUN: streamlit run app.py

MORSE_CODE_DICT = { 'A':'.-', 'B':'-...',
                    'C':'-.-.', 'D':'-..', 'E':'.',
                    'F':'..-.', 'G':'--.', 'H':'....',
                    'I':'..', 'J':'.---', 'K':'-.-',
                    'L':'.-..', 'M':'--', 'N':'-.',
                    'O':'---', 'P':'.--.', 'Q':'--.-',
                    'R':'.-.', 'S':'...', 'T':'-',
                    'U':'..-', 'V':'...-', 'W':'.--',
                    'X':'-..-', 'Y':'-.--', 'Z':'--..',
                    '1':'.----', '2':'..---', '3':'...--',
                    '4':'....-', '5':'.....', '6':'-....',
                    '7':'--...', '8':'---..', '9':'----.',
                    '0':'-----', ', ':'--..--', '.':'.-.-.-',
                    '?':'..--..', '/':'-..-.', '-':'-....-',
                    '(':'-.--.', ')':'-.--.-'}

def encrypt(message):
    cipher = ''
    for letter in message:
        if letter != ' ':
            cipher += MORSE_CODE_DICT[letter] + ' '

        else:
            cipher += ' '
 
    return cipher


def decrypt(message):
    message += ' '
    decipher = ''
    citext = ''
    for letter in message:
        if (letter != ' '):
            i = 0
            citext += letter
 
        else:
            i += 1
            if i == 2 :
                decipher += ' '

            else:
                decipher += list(MORSE_CODE_DICT.keys())[list(MORSE_CODE_DICT
                .values()).index(citext)]
                citext = ''
 
    return decipher


def file_upload(key):
    uploaded_file = st.file_uploader("Choose a file", key= key)
    if uploaded_file is not None:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        st.write(string_data)
        return string_data

def from_morse():
    st.markdown("# Frome MORSE")
    st.sidebar.markdown("# Frome MORSE")
    message = file_upload(key = 3)
    if message is not None:
        result = decrypt(message)
        st.write(result)


def to_morse():
    st.markdown("# To MORSE")
    st.sidebar.markdown("# To MORSE")
    message = file_upload(key = 2)
    if message is not None:
        result = encrypt(message.upper())
        st.write(result)


def to_hiragana():
    st.markdown("# To Hiragana :mount_fuji: from Latina :dancer:")
    st.sidebar.markdown("# To Hiragana :mount_fuji:")
    file_upload(key = 1)


def to_latina():
    st.markdown("# To Latina :dancer: from Hiragana :mount_fuji:")
    st.sidebar.markdown("# To Latina :dancer:")
    file_upload(key = 0)


def main_page():
    st.markdown("# Main page :page_facing_up:")
    st.sidebar.markdown("# Main page :page_facing_up:")


def main():
    page_names_to_funcs = {
    "Main": main_page,
    "To Latina": to_latina,
    "To Hiragana": to_hiragana,
    "To Morse": to_morse,
    "From Morse": from_morse
    }       
    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()
       


if __name__ == '__main__':
    main()