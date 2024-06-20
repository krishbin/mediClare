import pytesseract
from PIL import Image
from langchain_community.llms import Ollama
from deep_translator import GoogleTranslator

llm=Ollama(model="llama3")

def extract_text_from_image(image_path):
    image_path = 'report.png'
    img = Image.open(image_path)
    # Perform OCR on the image
    text = pytesseract.image_to_string(img)
    return text

def simplify_medical_report(image_path):
    ocr_data=extract_text_from_image(image_path)
    summarization_prompt=[f"""You are an intelligent clinical languge model.
                        Below is a snippet of patient's discharge summary. Extract the  hospital course and laboratory tests from this OCR data. Return result in 
                        json format with keys hospital course and laboratory tests.
                        [OCR Data Begin]
                        {ocr_data}
                        [OCR Data End]
                        """]
    summarization_response = llm.generate(summarization_prompt)
    summarization_output=summarization_response.generations[0][0].text
    simplification_prompt=[f"""Please explain the following medical summary into simple words so that normal person can understand it.
                            [Medical summary Begin]
                            {summarization_output}
                            [Medical summary End]
                            """]
    simplification_response = llm.generate(simplification_prompt)
    simplification_output=simplification_response.generations[0][0].text
    return simplification_output

def translate_to_another_language(text_to_translate,target_language='ne'):
    translator = GoogleTranslator(source='auto', target=target_language)
    translated_text = translator.translate(text_to_translate)
    return translated_text

image_path="report.png"
simplified_output=translate_to_another_language(simplify_medical_report(image_path))
print(simplified_output)


