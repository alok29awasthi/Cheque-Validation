from img_to_text import flag, account, amount2
from twilio.rest import Client

accountSid = "ACd691b44886ebb2e8751943b3cb4a3fa7"
authToken = "8825709af1ff826f18e74b455a8ca34f"
twilioClient = Client(accountSid, authToken) 
myTwilioNumber = "+19093154533"
destCellPhone = "+917355560329"

if flag==1:
    myMessage = twilioClient.messages.create(body = "The cheque has bounced for the account number: "+account[0:4]+"*****", from_=myTwilioNumber, to=destCellPhone)
else:
    myMessage = twilioClient.messages.create(body = "The cheque has been verified for the account number: "+account[0:4]+"***** for amount "+str(amount2)+"\nThank for using our services!\nProject by Neha and Priyadarshini", from_=myTwilioNumber, to=destCellPhone) 
