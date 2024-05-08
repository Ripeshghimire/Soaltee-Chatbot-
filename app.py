from fastapi import HTTPException
from fastapi import FastAPI, File, UploadFile, Depends, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse,StreamingResponse,JSONResponse
import numpy as np 
from modelhandler import load_model,get_similar_response
import spacy
origins = ["http://127.0.0.1:8000"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
responses = [
   " Hi! Welcome to the Soaltee Hotels and Resorts Ask me your queries about The Soaltee Hotels and Resorts.",
    "The Soaltee Kathmandu is located at Tahachal, Kathmandu, Nepal, with a phone number: +977-1-4273999 and email: info@soaltee.com.",
    "The brands of Soaltee Hotels are: 1. The Soaltee, Kathmandu 2. Soaltee Westend Premier, Nepalgunj 3. Soaltee Westland Resort, Chitwan 4. Soaltee Westend, Itahari.",
    "Services provided by Soaltee include: - Exceptional Experience - Best Price Guarantee - Bespoke Luxury - Exquisite Dining - Captivating Events - Immersive Wellness.",
    "Special offers include: - Happy Hour - Hookah Promotion - Mango Mania - Saturday Brunch (For each location, different offers are available.)",
    "To make a reservation, visit 'https://soaltee.com/' and fill out the form.",
    "Hotel rooms at Soaltee include: - Club Room - Executive Suite - Deluxe Room - Accessible Room - Classic Heritage - Executive Room - Regal Suite - Presidential Suite.",
    "Restaurants at Soaltee include: - Sunrise Restaurant (Lobby, serving breakfast, brunch, lunch, dinner, dessert) - Sunset Bar (Lobby, open from 10:00 AM to 10:45 PM) - Coffee Lounge (Lobby, open from 10:00 AM to 10:00 PM) - Golden Terrace - All Day Dining (Lobby Level) - Bao Xuan - Flavors of China (Lobby Level).",
    "Soaltee provides well-equipped conference rooms and health clubs with fitness centers, massages, steam, sauna, and mantra spas.",
    "The price of hotel rooms varies; check the reservation page for specific room rates.",
    "Breakfast and dinner are included in the hotel price.",
    "The Soaltee, a Nepali brand established in 1966, is a pioneer in the hospitality industry. They are known for their authentic and personalized cultural experiences, rooted in the country's rich heritage and delivered with heartfelt passion.",
    "The address of The Soaltee: Tahachal, Kathmandu, Nepal, 44600 Kathmandu, Nepal. Soaltee Westend Premier, Nepalgunj. Soaltee Westland Resort, Chitwan. Soaltee Westend, Itahari.",
    "Phone number for Soaltee Westend Premier Nepalgunj: +977–81-551145. Email: res.swp@soaltee.com.",
    "Address for Soaltee Westend Premier Nepalgunj: Bhujaigaun, Basudevpur, Nepalgunj, Nepal.", 
    "Phone number for Soaltee Westend Resort Chitwan: +977–56-411122. Email: res.swrc@soaltee.com.",
    "Address for Soaltee Westend Resort Chitwan: BMC – 22, Pathiani, Chitwan, Nepal.",
    "Phone number for Soaltee Westend Itahari: +977–25–590317/18/19. Email: info.swi@soaltee.com.",
    "Address for Soaltee Westend Itahari: Dharan Road, Itahari – 2, Sunsari 56705.",
    "You can store your luggage at the concierge.",
    "Tranquility Spa, located next to the swimming pool, is newly opened.",
    "Smoking is allowed only in the smoking rooms.",
    "Amenities not placed in the rooms can be requested and will be delivered. A list of these amenities can be found in the 'Forget Something?' card in the bathroom.",
    "A baby sitter can be arranged with prior information.",
    "Safe deposit boxes are available in all rooms and at the cashier counter, free of charge.",
    "An ATM machine is available on the premises, near the porch.",
    "Lost keys can be replaced by contacting the front desk or duty manager.",
    "The travel desk at the lobby can assist with sightseeing tours.",
    "Checkout time is 11 am; contact the duty manager for late checkout.",
    "City maps are available at the reception/concierge.",
    "Each room has direct dial facility; dial '9' for local calls and '9' followed by country code, city code, and phone number for international calls. Dial '0' for operator assistance.",
    "The duty manager will be sent for assistance.",
    "The laundry is at Soaltee Mode, a few yards from the hotel's main gate. Assistance can be provided if needed.",
    "The spa & jacuzzi are available at Tranquility Spa, located next to the swimming pool.",
    "Laptops can be used in the restaurants.",
    "The travel desk at the lobby can help reconfirm flight tickets.",
    "Postage stamps are available at the concierge desk.",
    "Bed linen is changed daily upon request or if soiled.",
    "Breakfast is served at the coffee shop 'Garden Terrace' at the lobby level.",
    "Contact the Duty Manager's desk at extn 41 for assistance.",
    "Complimentary newspapers are available in public areas and restaurants; assistance can be provided to obtain a copy.",
    "Restaurants at the hotel include Garden Terrace or Coffee Shop, Bao Xuan (serving Chinese cuisine), and Kakori (serving Indian cuisine).",
    "The nearest movie theater is at Civil Mall Shopping Mall.",
    "Local products can be found in Thamel.",
    "Places to visit near the hotel include Bhaktapur Durbar Square, Patan Durbar Square, Kathmandu Durbar Square, Swayambhunath Temple, Pashupatinath Temple, Chandragiri Hills.",
    "Phones can be charged at the reception.",
    "Photocopying and printouts can be done at the Business Center at the lobby level."
]
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')# initializing sentence transformer model
model = load_model()
embeddings = model.encode(responses) #encoding the response 
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
# ner = spacy.load()
@app.get('/',response_class=HTMLResponse)
async def root(request:Request):
    return templates.TemplateResponse(request=request,name= 'index.html')


@app.post('/question')
async def similarity(request:Request):
    '''expects a question from the user and finds the cosine similarity between the question and the response 
    in the response variable and returns answer based on the cosine similarity putting a threshold '''
    question = await request.json()
    question = question['question']
    # question  = model.encode(question)
    similar_position,highest_cossine_value = get_similar_response(question,embedding=embeddings,model=model)
    # similarity = cosine_similarity([question],embeddings).reshape(-1)
    # similar_position,similarity_value = np.argmax(similarity), similarity.max()
    similar_text = {'similar_text': responses[similar_position], 'cosine_similarity': float(highest_cossine_value)}
    if highest_cossine_value > 0.5:
        return JSONResponse(content = similar_text)
    else:
        return JSONResponse(content="ask similar question")
    
    


