import type { Intent, Entity, TrainingData } from "@/types";
import type { TrainedModel } from "./classifier";
import { type EnsembleModel, serializeEnsemble, deserializeEnsemble } from "./engine/ensemble";

const STORAGE_KEY = "nlu-bot-trainer-data";
const MODEL_KEY = "nlu-bot-trainer-model";
const ENSEMBLE_KEY = "nlu-ensemble-model-v2";

const INTENT_COLORS = [
  "#4c6ef5", "#7950f2", "#be4bdb", "#e64980", "#f76707",
  "#f59f00", "#40c057", "#15aabf", "#228be6", "#fa5252",
  "#845ef7", "#339af0", "#20c997", "#ff6b6b", "#fcc419",
];

const ENTITY_COLORS = [
  "#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c", "#38d9a9",
  "#3bc9db", "#4dabf7", "#748ffc", "#da77f2", "#f783ac",
];

function ex(id: string, text: string, intent: string) {
  return { id, text, intent, entities: [] as never[] };
}

function getDefaultData(): TrainingData {
  return {
    intents: [
      {
        id: "intent_greet", name: "greet",
        description: "Customer greets or initiates conversation",
        color: INTENT_COLORS[0],
        examples: [
          ex("g1","hello","greet"), ex("g2","hi there","greet"), ex("g3","good morning","greet"),
          ex("g4","hey","greet"), ex("g5","hi","greet"), ex("g6","good afternoon","greet"),
          ex("g7","good evening","greet"), ex("g8","howdy","greet"), ex("g9","what's up","greet"),
          ex("g10","greetings","greet"), ex("g11","hello there","greet"), ex("g12","hey there","greet"),
          ex("g13","hiya","greet"), ex("g14","yo","greet"), ex("g15","sup","greet"),
          ex("g16","good day","greet"), ex("g17","nice to meet you","greet"),
          ex("g18","pleased to meet you","greet"), ex("g19","morning","greet"), ex("g20","evening","greet"),
          ex("g21","hey how are you","greet"), ex("g22","hello is anyone there","greet"),
          ex("g23","hi I need some help","greet"), ex("g24","hey good to see you","greet"),
          ex("g25","aloha","greet"), ex("g26","well hello there","greet"),
          ex("g27","hi again","greet"), ex("g28","hey how's it going","greet"),
          ex("g29","good morning I have a question","greet"), ex("g30","afternoon","greet"),
          ex("g31","hey can someone help me","greet"), ex("g32","hello hello","greet"),
          ex("g33","oh hi","greet"), ex("g34","hey what's going on","greet"),
          ex("g35","hi I'm new here","greet"),
        ],
      },
      {
        id: "intent_goodbye", name: "goodbye",
        description: "Customer ends conversation",
        color: INTENT_COLORS[1],
        examples: [
          ex("b1","bye","goodbye"), ex("b2","goodbye","goodbye"), ex("b3","see you later","goodbye"),
          ex("b4","take care","goodbye"), ex("b5","see you","goodbye"), ex("b6","have a nice day","goodbye"),
          ex("b7","catch you later","goodbye"), ex("b8","talk to you soon","goodbye"),
          ex("b9","bye bye","goodbye"), ex("b10","good night","goodbye"), ex("b11","farewell","goodbye"),
          ex("b12","until next time","goodbye"), ex("b13","I'm leaving now","goodbye"),
          ex("b14","that's all I needed","goodbye"), ex("b15","gotta go","goodbye"),
          ex("b16","I'm done here","goodbye"), ex("b17","have a great day","goodbye"),
          ex("b18","peace out","goodbye"), ex("b19","later","goodbye"), ex("b20","ttyl","goodbye"),
          ex("b21","alright bye then","goodbye"), ex("b22","I'll be going now","goodbye"),
          ex("b23","thanks that's all bye","goodbye"), ex("b24","ok I'm off","goodbye"),
          ex("b25","see ya","goodbye"), ex("b26","signing off now","goodbye"),
          ex("b27","I have to go now goodbye","goodbye"), ex("b28","night night","goodbye"),
          ex("b29","cheers bye","goodbye"), ex("b30","ok thanks bye","goodbye"),
          ex("b31","that was everything I needed goodbye","goodbye"),
          ex("b32","cya later","goodbye"), ex("b33","adios","goodbye"),
          ex("b34","I'm heading out","goodbye"), ex("b35","toodles","goodbye"),
        ],
      },
      {
        id: "intent_thanks", name: "thank_you",
        description: "Customer expresses gratitude",
        color: INTENT_COLORS[2],
        examples: [
          ex("t1","thank you","thank_you"), ex("t2","thanks a lot","thank_you"),
          ex("t3","appreciate it","thank_you"), ex("t4","thanks so much","thank_you"),
          ex("t5","thanks","thank_you"), ex("t6","much appreciated","thank_you"),
          ex("t7","that's very helpful","thank_you"), ex("t8","you've been great","thank_you"),
          ex("t9","thank you very much","thank_you"), ex("t10","great help thanks","thank_you"),
          ex("t11","wonderful thank you","thank_you"), ex("t12","thanks for your help","thank_you"),
          ex("t13","I really appreciate your assistance","thank_you"),
          ex("t14","you're awesome thanks","thank_you"), ex("t15","perfect thanks","thank_you"),
          ex("t16","cheers mate","thank_you"), ex("t17","brilliant thank you","thank_you"),
          ex("t18","that was helpful","thank_you"), ex("t19","I appreciate that","thank_you"),
          ex("t20","thank you for the quick response","thank_you"),
          ex("t21","amazing service thank you","thank_you"),
          ex("t22","that solved my issue thanks","thank_you"),
          ex("t23","exactly what I needed thanks","thank_you"),
          ex("t24","you really helped me out","thank_you"),
          ex("t25","great job thanks so much","thank_you"),
          ex("t26","super helpful I appreciate it","thank_you"),
          ex("t27","thankful for your patience","thank_you"),
          ex("t28","that fixed it thank you","thank_you"),
          ex("t29","grateful for the assistance","thank_you"),
          ex("t30","nice one thanks","thank_you"),
          ex("t31","splendid work thank you","thank_you"),
          ex("t32","that was fast thanks","thank_you"),
          ex("t33","you made my day thanks","thank_you"),
          ex("t34","excellent support thank you","thank_you"),
          ex("t35","I'm satisfied with your help thanks","thank_you"),
        ],
      },
      {
        id: "intent_order_status", name: "order_status",
        description: "Customer asks about order status or tracking",
        color: INTENT_COLORS[3],
        examples: [
          ex("os1","where is my order","order_status"), ex("os2","track my package","order_status"),
          ex("os3","when will my order arrive","order_status"),
          ex("os4","what's the status of my delivery","order_status"),
          ex("os5","has my order shipped yet","order_status"),
          ex("os6","I want to know where my package is","order_status"),
          ex("os7","can I track my order","order_status"),
          ex("os8","my order hasn't arrived yet","order_status"),
          ex("os9","order tracking information","order_status"),
          ex("os10","when is my delivery expected","order_status"),
          ex("os11","check order status","order_status"),
          ex("os12","estimated delivery date","order_status"),
          ex("os13","give me tracking number","order_status"),
          ex("os14","is my package on the way","order_status"),
          ex("os15","my shipment is late","order_status"),
          ex("os16","delivery update please","order_status"),
          ex("os17","what's happening with my order","order_status"),
          ex("os18","order still processing","order_status"),
          ex("os19","any update on my package","order_status"),
          ex("os20","where's my stuff","order_status"),
          ex("os21","how long until my order arrives","order_status"),
          ex("os22","I placed an order three days ago where is it","order_status"),
          ex("os23","show me the shipping progress","order_status"),
          ex("os24","what carrier is delivering my package","order_status"),
          ex("os25","has my item been dispatched","order_status"),
          ex("os26","I need the tracking link for my order","order_status"),
          ex("os27","when was my order shipped","order_status"),
          ex("os28","is there a delay with my delivery","order_status"),
          ex("os29","my package seems stuck in transit","order_status"),
          ex("os30","what is the expected arrival date","order_status"),
          ex("os31","I haven't received my order yet","order_status"),
          ex("os32","order confirmation says shipped but no update","order_status"),
          ex("os33","how do I check my shipping status","order_status"),
          ex("os34","tell me when my order will be delivered","order_status"),
          ex("os35","I'm waiting for a package from you","order_status"),
        ],
      },
      {
        id: "intent_return", name: "return_product",
        description: "Customer wants to return a product",
        color: INTENT_COLORS[4],
        examples: [
          ex("r1","I want to return this item","return_product"),
          ex("r2","how do I return a product","return_product"),
          ex("r3","I need to send this back","return_product"),
          ex("r4","what is your return policy","return_product"),
          ex("r5","can I get a return label","return_product"),
          ex("r6","I'd like to return my purchase","return_product"),
          ex("r7","return this order","return_product"),
          ex("r8","I want to ship this item back to you","return_product"),
          ex("r9","initiate a return","return_product"),
          ex("r10","how many days do I have to return","return_product"),
          ex("r11","where do I drop off the return","return_product"),
          ex("r12","return shipping address","return_product"),
          ex("r13","is return shipping free","return_product"),
          ex("r14","I need a prepaid return label","return_product"),
          ex("r15","start the return process","return_product"),
          ex("r16","can I exchange this instead of returning","return_product"),
          ex("r17","return window for this product","return_product"),
          ex("r18","I changed my mind I want to send it back","return_product"),
          ex("r19","this doesn't fit I need to return it","return_product"),
          ex("r20","how to arrange a return pickup","return_product"),
          ex("r21","the item is the wrong size can I return it","return_product"),
          ex("r22","I want to swap this for a different one","return_product"),
          ex("r23","what are the conditions for returning items","return_product"),
          ex("r24","do I need the original packaging to return","return_product"),
          ex("r25","print a return slip for me","return_product"),
          ex("r26","I'd like an exchange not a return","return_product"),
          ex("r27","the product doesn't match the description I want to return it","return_product"),
          ex("r28","send me a return authorization number","return_product"),
          ex("r29","can I return after 30 days","return_product"),
          ex("r30","I received the wrong item I need to return it","return_product"),
          ex("r31","schedule a return pickup at my address","return_product"),
          ex("r32","where is the nearest drop off point for returns","return_product"),
          ex("r33","do you offer free return shipping","return_product"),
          ex("r34","I opened the package can I still return","return_product"),
          ex("r35","what happens after I ship my return","return_product"),
        ],
      },
      {
        id: "intent_refund", name: "refund_request",
        description: "Customer requests a refund or money back",
        color: INTENT_COLORS[5],
        examples: [
          ex("rf1","I want a refund","refund_request"),
          ex("rf2","can I get my money back","refund_request"),
          ex("rf3","please process my refund","refund_request"),
          ex("rf4","I need a refund for this order","refund_request"),
          ex("rf5","when will I receive my refund","refund_request"),
          ex("rf6","refund status","refund_request"),
          ex("rf7","I was charged incorrectly please refund me","refund_request"),
          ex("rf8","how long does a refund take","refund_request"),
          ex("rf9","give me my money back","refund_request"),
          ex("rf10","issue a refund please","refund_request"),
          ex("rf11","I want my money refunded to my card","refund_request"),
          ex("rf12","refund hasn't appeared in my bank account","refund_request"),
          ex("rf13","I'm still waiting for my refund","refund_request"),
          ex("rf14","credit my account back please","refund_request"),
          ex("rf15","where is my refund","refund_request"),
          ex("rf16","refund to my original payment method","refund_request"),
          ex("rf17","I was overcharged I need money back","refund_request"),
          ex("rf18","process the refund immediately","refund_request"),
          ex("rf19","how do I get a refund","refund_request"),
          ex("rf20","refund my purchase price","refund_request"),
          ex("rf21","I demand a full refund right now","refund_request"),
          ex("rf22","reimburse me for the defective product","refund_request"),
          ex("rf23","the refund should have arrived by now","refund_request"),
          ex("rf24","can I get store credit instead of a refund","refund_request"),
          ex("rf25","refund the difference in price","refund_request"),
          ex("rf26","I returned the item but no refund yet","refund_request"),
          ex("rf27","partial refund for damaged goods","refund_request"),
          ex("rf28","how many business days for the refund to process","refund_request"),
          ex("rf29","will the refund go back to my credit card","refund_request"),
          ex("rf30","I never received the item I want my money back","refund_request"),
          ex("rf31","the service was terrible I expect a refund","refund_request"),
          ex("rf32","do you have a money back guarantee","refund_request"),
          ex("rf33","I want to be reimbursed for shipping costs","refund_request"),
          ex("rf34","apply the refund to my next purchase","refund_request"),
          ex("rf35","why is my refund only partial","refund_request"),
        ],
      },
      {
        id: "intent_complaint", name: "complaint",
        description: "Customer files a complaint about quality or experience",
        color: INTENT_COLORS[6],
        examples: [
          ex("c1","I have a complaint","complaint"),
          ex("c2","this product is defective","complaint"),
          ex("c3","I received a damaged item","complaint"),
          ex("c4","the quality is terrible","complaint"),
          ex("c5","this is not what I ordered","complaint"),
          ex("c6","I'm very disappointed with this purchase","complaint"),
          ex("c7","the wrong item was delivered","complaint"),
          ex("c8","product arrived broken","complaint"),
          ex("c9","I want to file a complaint","complaint"),
          ex("c10","terrible experience with your service","complaint"),
          ex("c11","your product quality is awful","complaint"),
          ex("c12","this item doesn't work at all","complaint"),
          ex("c13","I am extremely unhappy","complaint"),
          ex("c14","this is unacceptable","complaint"),
          ex("c15","worst purchase I've ever made","complaint"),
          ex("c16","the package was damaged when it arrived","complaint"),
          ex("c17","missing parts in my order","complaint"),
          ex("c18","the color is completely different from the photo","complaint"),
          ex("c19","very poor quality product","complaint"),
          ex("c20","I am dissatisfied with my order","complaint"),
          ex("c21","the stitching is already coming apart","complaint"),
          ex("c22","I've had nothing but problems with this product","complaint"),
          ex("c23","the item stopped working after one day","complaint"),
          ex("c24","this looks nothing like the pictures on your website","complaint"),
          ex("c25","there was a scratch on the product out of the box","complaint"),
          ex("c26","I'm furious about the poor quality","complaint"),
          ex("c27","the food was stale when it arrived","complaint"),
          ex("c28","your packaging is horrible everything was crushed","complaint"),
          ex("c29","I ordered two but only got one and it was broken","complaint"),
          ex("c30","the instructions are missing and the product is faulty","complaint"),
          ex("c31","I'm really upset about the service I got","complaint"),
          ex("c32","the expiration date had already passed","complaint"),
          ex("c33","this is a scam the product is fake","complaint"),
          ex("c34","I want to report a quality issue","complaint"),
          ex("c35","the screen has dead pixels right out of the box","complaint"),
        ],
      },
      {
        id: "intent_product_info", name: "product_inquiry",
        description: "Customer asks about product details, specs, or availability",
        color: INTENT_COLORS[7],
        examples: [
          ex("p1","do you have this in stock","product_inquiry"),
          ex("p2","what sizes are available","product_inquiry"),
          ex("p3","tell me about this product","product_inquiry"),
          ex("p4","is this item available in blue","product_inquiry"),
          ex("p5","what are the specifications","product_inquiry"),
          ex("p6","how much does this cost","product_inquiry"),
          ex("p7","is this compatible with my device","product_inquiry"),
          ex("p8","what material is this made of","product_inquiry"),
          ex("p9","do you ship internationally","product_inquiry"),
          ex("p10","when will this be back in stock","product_inquiry"),
          ex("p11","what colors does this come in","product_inquiry"),
          ex("p12","what's the price of this item","product_inquiry"),
          ex("p13","is there a warranty","product_inquiry"),
          ex("p14","product dimensions please","product_inquiry"),
          ex("p15","does this come with batteries","product_inquiry"),
          ex("p16","what's the weight of this product","product_inquiry"),
          ex("p17","is this waterproof","product_inquiry"),
          ex("p18","compare this with the pro version","product_inquiry"),
          ex("p19","any discounts available on this","product_inquiry"),
          ex("p20","tell me the features of this item","product_inquiry"),
          ex("p21","how many units are left in stock","product_inquiry"),
          ex("p22","what is the battery life on this device","product_inquiry"),
          ex("p23","is there a newer model coming out soon","product_inquiry"),
          ex("p24","can you tell me about the warranty coverage","product_inquiry"),
          ex("p25","does this product come in other colors","product_inquiry"),
          ex("p26","what are the system requirements","product_inquiry"),
          ex("p27","show me similar products in this category","product_inquiry"),
          ex("p28","is this product suitable for kids","product_inquiry"),
          ex("p29","do you have a size guide for this item","product_inquiry"),
          ex("p30","what is the resolution of this display","product_inquiry"),
          ex("p31","how much storage does this come with","product_inquiry"),
          ex("p32","is assembly required for this item","product_inquiry"),
          ex("p33","can I see customer reviews for this product","product_inquiry"),
          ex("p34","what are the available configurations","product_inquiry"),
          ex("p35","does this product work with bluetooth","product_inquiry"),
        ],
      },
      {
        id: "intent_cancel", name: "cancel_order",
        description: "Customer wants to cancel an order before delivery",
        color: INTENT_COLORS[8],
        examples: [
          ex("co1","I want to cancel my order","cancel_order"),
          ex("co2","please cancel this order","cancel_order"),
          ex("co3","can I cancel my purchase","cancel_order"),
          ex("co4","cancel order number","cancel_order"),
          ex("co5","I changed my mind I want to cancel","cancel_order"),
          ex("co6","how do I cancel an order","cancel_order"),
          ex("co7","stop my order from shipping","cancel_order"),
          ex("co8","I don't want this anymore please cancel","cancel_order"),
          ex("co9","cancel my recent purchase","cancel_order"),
          ex("co10","is it too late to cancel","cancel_order"),
          ex("co11","void my order","cancel_order"),
          ex("co12","I placed the wrong order please cancel","cancel_order"),
          ex("co13","cancel before it ships","cancel_order"),
          ex("co14","I need to cancel right away","cancel_order"),
          ex("co15","revoke my order","cancel_order"),
          ex("co16","I accidentally ordered this please cancel it","cancel_order"),
          ex("co17","withdrawal my order","cancel_order"),
          ex("co18","don't ship my order cancel it","cancel_order"),
          ex("co19","I no longer need this cancel the order","cancel_order"),
          ex("co20","cancellation request","cancel_order"),
          ex("co21","undo my last order please","cancel_order"),
          ex("co22","I decided not to buy this anymore cancel it","cancel_order"),
          ex("co23","can I still cancel if it hasn't shipped","cancel_order"),
          ex("co24","please stop processing my order","cancel_order"),
          ex("co25","I ordered by mistake please cancel the order","cancel_order"),
          ex("co26","remove my order from the system","cancel_order"),
          ex("co27","halt the shipment and cancel","cancel_order"),
          ex("co28","I found it cheaper elsewhere please cancel","cancel_order"),
          ex("co29","cancel all pending orders on my account","cancel_order"),
          ex("co30","I want to cancel and reorder with a different address","cancel_order"),
          ex("co31","abort this purchase","cancel_order"),
          ex("co32","is there a cancellation fee","cancel_order"),
          ex("co33","I need to cancel within the cancellation window","cancel_order"),
          ex("co34","scratch that order I don't want it","cancel_order"),
          ex("co35","please void my recent transaction","cancel_order"),
        ],
      },
      {
        id: "intent_payment", name: "payment_issue",
        description: "Customer has payment, billing, or transaction problems",
        color: INTENT_COLORS[9],
        examples: [
          ex("py1","my payment failed","payment_issue"),
          ex("py2","I was double charged","payment_issue"),
          ex("py3","payment not going through","payment_issue"),
          ex("py4","card declined when trying to pay","payment_issue"),
          ex("py5","billing issue with my account","payment_issue"),
          ex("py6","I see an unauthorized charge","payment_issue"),
          ex("py7","what payment methods do you accept","payment_issue"),
          ex("py8","can I pay with PayPal","payment_issue"),
          ex("py9","my transaction is pending","payment_issue"),
          ex("py10","update my payment information","payment_issue"),
          ex("py11","credit card not working on your site","payment_issue"),
          ex("py12","charged twice for the same order","payment_issue"),
          ex("py13","payment declined error","payment_issue"),
          ex("py14","why was I charged extra","payment_issue"),
          ex("py15","incorrect amount on my credit card statement","payment_issue"),
          ex("py16","change my billing address","payment_issue"),
          ex("py17","payment confirmation not received","payment_issue"),
          ex("py18","do you accept Apple Pay","payment_issue"),
          ex("py19","invoice for my order","payment_issue"),
          ex("py20","my coupon code isn't working at checkout","payment_issue"),
          ex("py21","the promo code gives an error at payment","payment_issue"),
          ex("py22","I got charged but the order didn't go through","payment_issue"),
          ex("py23","my bank says the transaction was blocked","payment_issue"),
          ex("py24","can I split the payment between two cards","payment_issue"),
          ex("py25","do you offer installment payment plans","payment_issue"),
          ex("py26","the checkout page keeps crashing when I try to pay","payment_issue"),
          ex("py27","I need a receipt for tax purposes","payment_issue"),
          ex("py28","there's a pending charge that shouldn't be there","payment_issue"),
          ex("py29","can I change the card on file for my subscription","payment_issue"),
          ex("py30","the total at checkout doesn't match the listed price","payment_issue"),
          ex("py31","I was charged sales tax but I'm tax exempt","payment_issue"),
          ex("py32","my gift card balance isn't showing at checkout","payment_issue"),
          ex("py33","payment security verification keeps failing","payment_issue"),
          ex("py34","do you accept cryptocurrency as payment","payment_issue"),
          ex("py35","I need to update the expiry date on my saved card","payment_issue"),
        ],
      },
      {
        id: "intent_account", name: "account_help",
        description: "Customer needs help with login, password, or account settings",
        color: INTENT_COLORS[10],
        examples: [
          ex("a1","I can't log in to my account","account_help"),
          ex("a2","reset my password","account_help"),
          ex("a3","how do I change my email address","account_help"),
          ex("a4","I forgot my password","account_help"),
          ex("a5","update my account details","account_help"),
          ex("a6","delete my account","account_help"),
          ex("a7","I want to change my shipping address on my profile","account_help"),
          ex("a8","how do I create an account","account_help"),
          ex("a9","account settings page not loading","account_help"),
          ex("a10","my account is locked","account_help"),
          ex("a11","sign up for a new account","account_help"),
          ex("a12","verification email not received","account_help"),
          ex("a13","change my username","account_help"),
          ex("a14","two factor authentication setup","account_help"),
          ex("a15","unsubscribe from emails","account_help"),
          ex("a16","my login credentials don't work","account_help"),
          ex("a17","how to update my phone number on profile","account_help"),
          ex("a18","deactivate my account temporarily","account_help"),
          ex("a19","password reset link expired","account_help"),
          ex("a20","manage my notification preferences","account_help"),
          ex("a21","I keep getting logged out automatically","account_help"),
          ex("a22","how do I merge two accounts","account_help"),
          ex("a23","my account was hacked what do I do","account_help"),
          ex("a24","I want to change the name on my account","account_help"),
          ex("a25","enable two step verification for security","account_help"),
          ex("a26","the sign up page won't load","account_help"),
          ex("a27","how do I view my order history in my account","account_help"),
          ex("a28","I never got the verification code","account_help"),
          ex("a29","can I have multiple accounts with the same email","account_help"),
          ex("a30","my profile picture won't upload","account_help"),
          ex("a31","I want to close my account permanently","account_help"),
          ex("a32","change the language settings on my account","account_help"),
          ex("a33","I can't remember my username or email","account_help"),
          ex("a34","link my social media account to my profile","account_help"),
          ex("a35","how do I download my account data","account_help"),
        ],
      },
      {
        id: "intent_human", name: "speak_to_human",
        description: "Customer wants to escalate to a live agent",
        color: INTENT_COLORS[11],
        examples: [
          ex("h1","I want to speak to a human","speak_to_human"),
          ex("h2","connect me to an agent","speak_to_human"),
          ex("h3","let me talk to a real person","speak_to_human"),
          ex("h4","transfer me to customer support","speak_to_human"),
          ex("h5","I need a human representative","speak_to_human"),
          ex("h6","can I speak with a manager","speak_to_human"),
          ex("h7","get me a live agent","speak_to_human"),
          ex("h8","I don't want to talk to a bot","speak_to_human"),
          ex("h9","put me through to someone","speak_to_human"),
          ex("h10","I want real support not automated","speak_to_human"),
          ex("h11","escalate this to a supervisor","speak_to_human"),
          ex("h12","is there a human I can chat with","speak_to_human"),
          ex("h13","this bot isn't helping connect me to staff","speak_to_human"),
          ex("h14","I need to speak to a person not a machine","speak_to_human"),
          ex("h15","live chat with a representative","speak_to_human"),
          ex("h16","operator please","speak_to_human"),
          ex("h17","I want to call your support team","speak_to_human"),
          ex("h18","redirect me to human support","speak_to_human"),
          ex("h19","hand me over to an actual person","speak_to_human"),
          ex("h20","enough with the bot let me talk to someone","speak_to_human"),
          ex("h21","can you transfer me to a live representative","speak_to_human"),
          ex("h22","I'd rather speak with a real agent","speak_to_human"),
          ex("h23","your automated system isn't resolving my issue","speak_to_human"),
          ex("h24","give me a phone number to reach a person","speak_to_human"),
          ex("h25","I insist on talking to a real human being","speak_to_human"),
          ex("h26","please forward my case to a senior agent","speak_to_human"),
          ex("h27","is there a way to email a real person","speak_to_human"),
          ex("h28","the chatbot can't help me I need a human","speak_to_human"),
          ex("h29","I've been going in circles connect me to staff","speak_to_human"),
          ex("h30","when is your live support available","speak_to_human"),
          ex("h31","I need someone who can actually fix this problem","speak_to_human"),
          ex("h32","are your human agents available right now","speak_to_human"),
          ex("h33","route me to the complaints department","speak_to_human"),
          ex("h34","stop giving me automated responses get me a person","speak_to_human"),
          ex("h35","I want to file a formal complaint with a manager","speak_to_human"),
        ],
      },
    ],
    entities: [
      {
        id: "entity_product",
        name: "product",
        description: "Product names or categories",
        values: ["laptop", "phone", "tablet", "headphones", "charger", "keyboard", "mouse", "monitor", "camera", "speaker"],
        color: ENTITY_COLORS[0],
      },
      {
        id: "entity_color",
        name: "color",
        description: "Product color options",
        values: ["red", "blue", "black", "white", "green", "silver", "gold", "pink"],
        color: ENTITY_COLORS[1],
      },
      {
        id: "entity_size",
        name: "size",
        description: "Size options",
        values: ["small", "medium", "large", "extra large", "XS", "XL"],
        color: ENTITY_COLORS[2],
      },
    ],
    metadata: {
      trainedAt: null,
      totalExamples: 420,
      version: "1.0.0",
    },
  };
}

export function loadData(): TrainingData {
  if (typeof window === "undefined") return getDefaultData();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return getDefaultData();
    const parsed = JSON.parse(raw);
    // Reset if old dataset (fewer examples per intent)
    if (parsed.intents && parsed.metadata?.totalExamples < 400) return getDefaultData();
    return parsed;
  } catch {
    return getDefaultData();
  }
}

export function saveData(data: TrainingData): void {
  if (typeof window === "undefined") return;
  data.metadata.totalExamples = data.intents.reduce(
    (sum, i) => sum + i.examples.length, 0
  );
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch (e) {
    if (e instanceof DOMException && e.name === "QuotaExceededError") {
      // Clear old single-classifier model to make room for training data
      localStorage.removeItem(MODEL_KEY);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } else {
      throw e;
    }
  }
}

export function loadModel(): TrainedModel | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(MODEL_KEY);
    if (!raw) return null;
    const model = JSON.parse(raw);
    // Reject models from incompatible tokenizer versions
    if (!model.version || model.version < 5) return null;
    return model;
  } catch {
    return null;
  }
}

export function saveModel(model: TrainedModel): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(MODEL_KEY, JSON.stringify(model));
}

export function clearModel(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(MODEL_KEY);
  localStorage.removeItem(ENSEMBLE_KEY);
}

export function loadEnsembleModel(): EnsembleModel | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(ENSEMBLE_KEY);
    if (!raw) return null;
    return deserializeEnsemble(raw);
  } catch {
    return null;
  }
}

export function saveEnsembleModel(model: EnsembleModel): void {
  if (typeof window === "undefined") return;
  // Clear old active model keys to free space
  localStorage.removeItem(ENSEMBLE_KEY);
  localStorage.removeItem(MODEL_KEY);
  // Note: we do NOT evict nlu-model-v-* keys here — those belong to the
  // model registry and are needed for version rollback. If localStorage
  // quota is exceeded, the registry's own eviction logic should handle it.
  try {
    localStorage.setItem(ENSEMBLE_KEY, serializeEnsemble(model));
  } catch (e) {
    // If quota exceeded, evict oldest version artifacts as a last resort
    if (e instanceof DOMException && e.name === "QuotaExceededError") {
      const versionKeys: string[] = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key?.startsWith("nlu-model-v-")) versionKeys.push(key);
      }
      // Remove oldest versions first (sorted lexicographically, oldest = lowest semver)
      versionKeys.sort();
      for (const key of versionKeys) {
        localStorage.removeItem(key);
        try {
          localStorage.setItem(ENSEMBLE_KEY, serializeEnsemble(model));
          return; // success after eviction
        } catch {
          continue; // still not enough space, evict next
        }
      }
      throw e; // re-throw if we still can't save after evicting everything
    }
    throw e;
  }
}

export function clearEnsembleModel(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(ENSEMBLE_KEY);
}

export function getNextIntentColor(intents: Intent[]): string {
  return INTENT_COLORS[intents.length % INTENT_COLORS.length];
}

export function getNextEntityColor(entities: Entity[]): string {
  return ENTITY_COLORS[entities.length % ENTITY_COLORS.length];
}

export function exportRasaFormat(data: TrainingData): object {
  return {
    version: "3.1",
    nlu: data.intents.map((intent) => ({
      intent: intent.name,
      examples: intent.examples.map((ex) => {
        let text = ex.text;
        const sorted = [...ex.entities].sort((a, b) => b.start - a.start);
        for (const entity of sorted) {
          text = text.slice(0, entity.start) + `[${entity.value}](${entity.entity})` + text.slice(entity.end);
        }
        return `- ${text}`;
      }).join("\n"),
    })),
  };
}

export function exportJsonFormat(data: TrainingData): object {
  return {
    intents: data.intents.map((i) => ({
      name: i.name,
      description: i.description,
      examples: i.examples.map((e) => ({ text: e.text, entities: e.entities })),
    })),
    entities: data.entities.map((e) => ({
      name: e.name,
      description: e.description,
      values: e.values,
    })),
  };
}
