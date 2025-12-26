

"""

Streamlit app for Facial Emotion-Driven Review Analysis.

This version includes extensive debugging and error handling..

"""



import streamlit as st 

from PIL import Image 

import time 

import traceback 

import sys 

import os 





print ("="*80 )

print ("DEBUG: STREAMLIT APP STARTING")

print (f"DEBUG: Python version: {sys .version }")

print (f"DEBUG: Current working directory: {os .getcwd ()}")

print (f"DEBUG: Script location: {__file__ if '__file__'in globals ()else 'interactive'}")

print ("="*80 )





# try :

#     print ("DEBUG: Attempting to import model_utils...")

#     import model_utils 

#     print ("DEBUG: ✓ Successfully imported model_utils")

# except ImportError as e :

#     print (f"DEBUG: ✗ Failed to import model_utils: {e }")

#     print (f"DEBUG: Traceback: {traceback .format_exc ()}")

#     st .error (f"Failed to import model_utils: {e }")

#     st .stop ()


try:
    print("DEBUG: Attempting to import model_utils...")
    from core import models as model_utils
    print("DEBUG: ✓ Successfully imported model_utils")
except Exception as e:
    print(f"DEBUG: ✗ Failed to import model_utils: {e }")
    print(f"DEBUG: Traceback: {traceback.format_exc()}")
    st.error(f"Failed to import model_utils: {e }")
    st.stop()



st .set_page_config (

page_title ="Facial Emotion RAG System",

page_icon ="🤖",

layout ="wide"

)

## st .set_page_config (

## page_title ="Facial Emotion RAG System",

## page_icon ="🤖",

## layout ="wide"

# )



st .title ("Facial Emotion-Driven Review Analysis 🤖")

st .write ("This app combines two AI systems. Upload a photo to see a facial emotion, and the system will automatically query a review database for that emotion.")





with st .sidebar :

    st .header ("Debug Information")

    st .code (f"Python: {sys .version [:20 ]}...")

    st .code (f"Working Dir: {os .getcwd ()}")

   ## st .header ("Debug Information")

   ## st .code (f"Python: {sys .version [:30 ]}...")

   ## st .code (f"Working Dir: {os .getcwd ()}")




    if st .button ("Show Full Debug Info"):

        with st .expander ("System Information"):

            st .json ({

            "python_version":sys .version ,

            "working_directory":os .getcwd (),

            "streamlit_version":st .__version__ ,

            "torch_available":"torch"in sys .modules ,

            "transformers_available":"transformers"in sys .modules ,

            "langchain_available":"langchain"in sys .modules ,

            })



print ("DEBUG: Page configuration complete")





@st .cache_resource (show_spinner =False )

def load_all_models ():

    """Load all models with comprehensive error handling."""

    print ("DEBUG: === load_all_models START ===")



    errors =[]

    components ={}





    try :

        print ("DEBUG: Loading emotion model...")

        emotion_processor ,emotion_model ,emotion_device =model_utils .load_emotion_model ()

        components ['emotion']=(emotion_processor ,emotion_model ,emotion_device )

        print ("DEBUG: ✓ Emotion model loaded")

    except Exception as e :

        error_msg =f"Failed to load emotion model: {e }"

        print (f"DEBUG: ✗ {error_msg }")

        errors .append (error_msg )

        components ['emotion']=None 





    try :

        print ("DEBUG: Loading RAG pipeline...")

        db ,llm ,embeddings ,sentiment_pipe ,sentiment_labels =model_utils .load_rag_pipeline ()



        qa_chain =(db ,llm ,embeddings )

        components ['rag']=(qa_chain ,sentiment_pipe ,sentiment_labels )

        print ("DEBUG: ✓ RAG pipeline loaded")

    except Exception as e :

        error_msg =f"Failed to load RAG pipeline: {e }"

        print (f"DEBUG: ✗ {error_msg }")

        errors .append (error_msg )

        components ['rag']=None 



    print ("DEBUG: === load_all_models END ===")

    return components ,errors 





with st .spinner ("🚀 Warming up the AI models... This may take a moment the first time."):

    start_time =time .time ()



    try :

        components ,load_errors =load_all_models ()

        end_time =time .time ()



        if load_errors :

            st .warning (f"⚠️ Some components failed to load:")

            for error in load_errors :

                st .error (error )





            with st .expander ("Show Full Error Details"):

                st .code (traceback .format_exc ())





        if components .get ('emotion')and components .get ('rag'):

            st .success (f"✅ AI Models are ready! (Loaded in {end_time -start_time :.2f}s)")





            emotion_processor ,emotion_model ,emotion_device =components ['emotion']

            qa_chain ,sentiment_pipe ,sentiment_labels =components ['rag']



            print ("DEBUG: All models loaded and unpacked successfully")

        else :

            st .error ("❌ Critical components failed to load. Please check the errors above.")

            st .stop ()



    except Exception as e :

        print (f"DEBUG: CRITICAL ERROR during model loading: {e }")

        st .error (f"Critical error during initialization: {e }")





        st .subheader ("Full Error Traceback")

        error_traceback =traceback .format_exc ()

        st .code (error_traceback )

        print (f"DEBUG: Full traceback:\n{error_traceback }")



        st .error ("Please check:")

        st .markdown ("""

        1. ✅ Your Hugging Face repository names in `model_utils.py` are correct

        2. ✅ The repositories are public

        3. ✅ All required packages are installed (check requirements.txt)

        4. ✅ Your internet connection is stable

        """)

        st .stop ()





tab1 ,tab2 ,tab3 =st .tabs (["📸 Query with Photo","💬 Manual Query","🔍 Debug Tools"])

print ("DEBUG: Creating application tabs")





with tab1 :

    st .header ("Upload a Photo → Detect Emotion → Query Reviews")



    col1 ,col2 =st .columns ([1 ,1 ])



    with col1 :

        uploaded_file =st .file_uploader (

        "Choose a photo of a face...",

        type =["jpg","jpeg","png"],

        help ="Upload a clear photo of a face for emotion detection"

        )



        if uploaded_file is not None :

            print (f"DEBUG: Image uploaded: {uploaded_file .name }")

            image =Image .open (uploaded_file )

            st .image (image ,caption ="Uploaded Image",use_container_width =True )



    with col2 :

        if uploaded_file is not None :

            try :



                with st .spinner ("🔍 Analyzing facial emotion..."):

                    emotion =model_utils .predict_emotion (

                    emotion_processor ,emotion_model ,emotion_device ,image 

                    )



                st .success (f"**Detected Emotion:** {emotion .upper ()} 😊")

                print (f"DEBUG: Detected emotion: {emotion }")





                dynamic_query =f"What are customers saying who feel {emotion }?"

                st .info (f"**Dynamic Query:** *'{dynamic_query }'*")





                with st .spinner (f"📚 Searching review database for '{emotion }' reviews..."):

                    result =model_utils .query_rag (qa_chain ,dynamic_query )





                st .subheader ("🔍 AI Summary")

                st .write (result ['result'])



                if result ['source_documents']:

                    st .subheader ("📚 Retrieved Reviews")

                    for i ,doc in enumerate (result ['source_documents'],1 ):

                        sentiment =model_utils .analyze_sentiment (

                        sentiment_pipe ,sentiment_labels ,doc .page_content 

                        )



                        with st .container (border =True ):

                            col_a ,col_b =st .columns ([3 ,1 ])

                            with col_a :

                                st .markdown (f"**Review #{i }**")

                                emotion_tag =doc .metadata .get ('emotion','Unknown')

                                st .markdown (f"*Emotion Tag: {emotion_tag }*")

                            with col_b :

                                st .markdown (f"**Sentiment**")

                                st .markdown (sentiment )



                            st .write (doc .page_content )

                else :

                    st .warning ("No reviews found for this emotion.")



            except Exception as e :

                st .error (f"Error processing image: {e }")

                with st .expander ("Show Error Details"):

                    st .code (traceback .format_exc ())





with tab2 :

    st .header ("Query the Review Database Manually")





    text_query =st .text_input (

    "Enter your query:",

    placeholder ="e.g., 'What are customers happy about?'",

    help ="Ask any question about customer reviews"

    )





    st .markdown ("**Example queries:**")

    example_cols =st .columns (3 )

    example_queries =[

    "What are customers happy about?",

    "Show me angry customer reviews",

    "What makes customers sad?"

    ]



    for col ,example in zip (example_cols ,example_queries ):

        with col :

            if st .button (f"📝 {example }",key =f"example_{example }"):

                text_query =example 





    if st .button ("🔍 Search Reviews",type ="primary"):

        print (f"DEBUG: Manual search initiated with query: {text_query }")



        if text_query :

            try :

                with st .spinner ("Searching review database..."):

                    result =model_utils .query_rag (qa_chain ,text_query )





                st .subheader ("🔍 AI Summary")

                st .write (result ['result'])



                if result ['source_documents']:

                    st .subheader ("📚 Retrieved Reviews")

                    for i ,doc in enumerate (result ['source_documents'],1 ):

                        sentiment =model_utils .analyze_sentiment (

                        sentiment_pipe ,sentiment_labels ,doc .page_content 

                        )



                        with st .container (border =True ):

                            col_a ,col_b =st .columns ([3 ,1 ])

                            with col_a :

                                st .markdown (f"**Review #{i }**")

                                emotion_tag =doc .metadata .get ('emotion','Unknown')

                                st .markdown (f"*Emotion Tag: {emotion_tag }*")

                            with col_b :

                                st .markdown (f"**Sentiment**")

                                st .markdown (sentiment )



                            st .write (doc .page_content )

                else :

                    st .warning ("No reviews found matching your query.")



            except Exception as e :

                st .error (f"Error processing query: {e }")

                with st .expander ("Show Error Details"):

                    st .code (traceback .format_exc ())

        else :

            st .warning ("Please enter a query.")





with tab3 :

    st .header ("🔍 Debug Tools & System Information")



    col1 ,col2 =st .columns (2 )



    with col1 :

        st .subheader ("Component Status")





        components_status ={

        "Emotion Model":"✅ Loaded"if components .get ('emotion')else "❌ Failed",

        "RAG System":"✅ Loaded"if components .get ('rag')else "❌ Failed",

        "Vector Database":"✅ Ready"if components .get ('rag')else "❌ Not Available",

        "LLM Pipeline":"✅ Ready"if components .get ('rag')else "❌ Not Available",

        "Sentiment Analysis":"✅ Ready"if components .get ('rag')else "❌ Not Available"

        }



        for component ,status in components_status .items ():

            st .write (f"{component }: {status }")



    with col2 :

        st .subheader ("Quick Tests")



        if st .button ("Test Emotion Model"):

            try :



                test_img =Image .new ('RGB',(224 ,224 ),color ='white')

                emotion =model_utils .predict_emotion (

                emotion_processor ,emotion_model ,emotion_device ,test_img 

                )

                st .success (f"Test successful! Detected: {emotion }")

            except Exception as e :

                st .error (f"Test failed: {e }")



        if st .button ("Test RAG System"):

            try :

                result =model_utils .query_rag (qa_chain ,"test query")

                st .success (f"Test successful! Response length: {len (result ['result'])} chars")

            except Exception as e :

                st .error (f"Test failed: {e }")



        if st .button ("Test Sentiment Analysis"):

            try :

                sentiment =model_utils .analyze_sentiment (

                sentiment_pipe ,sentiment_labels ,"This is a great product!"

                )

                st .success (f"Test successful! Sentiment: {sentiment }")

            except Exception as e :

                st .error (f"Test failed: {e }")



print ("DEBUG: Streamlit app setup complete and running")







