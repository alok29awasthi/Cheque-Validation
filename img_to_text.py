import pytesseract       
from PIL import Image     
from twilio.rest import Client 

accountSid = "ACd691b44886ebb2e8751943b3cb4a3fa7"
authToken = "8825709af1ff826f18e74b455a8ca34f"
twilioClient = Client(accountSid, authToken) 
myTwilioNumber = "+19093154533"
destCellPhone = "+917355560329"

##variables
amount1=0
amount2=0
account=""
x=0
k=1

amt_word_1=Image.open('amt_word_1.jpg')
amt_word_2=Image.open('amt_word_2.jpg')
amt_num=Image.open('amt_num.jpg')
acc_num=Image.open('acc_num.jpg')

pytesseract.tesseract_cmd =r'C:\Program Files\Tesseract-OCR\tesseract'   

amt_word_1 = pytesseract.image_to_string(amt_word_1) 
amt_word_2 = pytesseract.image_to_string(amt_word_2) 
amt_num = pytesseract.image_to_string(amt_num) 
acc_num = pytesseract.image_to_string(acc_num)

##print(amt_word_1)
##print(amt_word_2)

with open('abc.txt',mode ='w') as file:      
                 file.write(amt_word_1)      
                 file.write(amt_word_2)      
                 file.write(amt_num)      
                 file.write(acc_num)

amt_word_1=amt_word_1.split(' ')
amt_word_2=amt_word_2.split(' ')
amt_word_1+=amt_word_2

for i in range(0,len(amt_word_1)):
    if(amt_word_1[i]=='ONE'):
        x+=1
    if(amt_word_1[i]=='TWO'):
        x+=2
    if(amt_word_1[i]=='THREE'):
        x+=3
    if(amt_word_1[i]=='FOUR'):
        x+=4
    if(amt_word_1[i]=='FIVE'):
        x+=5
    if(amt_word_1[i]=='SIX'):
        x+=6
    if(amt_word_1[i]=='SEVEN'):
        x+=7
    if(amt_word_1[i]=='EIGHT'):
        x+=8
    if(amt_word_1[i]=='NINE'):
        x+=9
    if(amt_word_1[i]=='TEN'):
        x+=10
    if(amt_word_1[i]=='ELEVEN'):
        x+=11
    if(amt_word_1[i]=='TWELVE'):
        x+=12
    if(amt_word_1[i]=='THIRTEEN'):
        x+=13
    if(amt_word_1[i]=='FOURTEEN'):
        x+=14
    if(amt_word_1[i]=='FIFTEEN'):
        x+=15
    if(amt_word_1[i]=='SIXTEEN'):
        x+=16
    if(amt_word_1[i]=='SEVENTEEN'):
        x+=17
    if(amt_word_1[i]=='EIGHTEEN'):
        x+=18
    if(amt_word_1[i]=='NINETEEN'):
        x+=19
    if(amt_word_1[i]=='TWENTY'):
        x+=20
    if(amt_word_1[i]=='THIRTY'):
        x+=30
    if(amt_word_1[i]=='FORTY'):
        x+=40
    if(amt_word_1[i]=='FIFTY'):
        x+=50
    if(amt_word_1[i]=='SIXTY'):
        x+=60
    if(amt_word_1[i]=='SEVENTY'):
        x+=70
    if(amt_word_1[i]=='EIGHTY'):
        x+=80
    if(amt_word_1[i]=='NINETY'):
        x+=90
    if(amt_word_1[i]=='HUNDRED'):
        x=x*100
        amount1+=x
        x=0
    if(amt_word_1[i]=='THOUSAND'):
        x=x*1000
        amount1+=x
        x=0
    if(amt_word_1[i]=='LAKH'):
        x=x*100000
        amount1+=x
        x=0
    if(amt_word_1[i]=='CRORE'):
        x=x*10000000
        amount1+=x
        x=0
amount1+=x

for i in range(len(amt_word_1)-1, -1, -1):
    if(amt_num[i].isdigit()==True):
        amount2+=(int(amt_num[i])*k)
        k=k*10

for i in range(0,len(acc_num)):
    if(acc_num[i].isdigit()==True):
        account+=acc_num[i]

flag=0

print("Amount in numbers: ",amount2)
print("Amount in words converted to numbers: ",amount1)        
print("Amount in words: "," ".join(amt_word_1))
print("Account number: ",account)

if(amount1!=amount2):
    print("\nThe cheque has bounced for the account number: "+account[0:4]+"*****")
    flag=1
else:
    print("\nThe cheque has been verified for the account number: "+account[0:4]+"***** for amount "+str(amount2)+"\nThank for using our services!\nProject by Neha and Priyadarshini")
