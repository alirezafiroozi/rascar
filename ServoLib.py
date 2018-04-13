import time
import RPi.GPIO as GPIO


#------------PWM initiation

motorFreq = 150 # in Hz
servoFreq = 100 # in Hz
motorPin = 21 #BCM pin number
servoPin = 18 #BCM pin number
motorStartDuty = 15 # Max duty cycle for Motor
servoStartDuty = 17.5 # Max duty cycle for Servo
GPIO.setmode(GPIO.BCM) #sets the GPIO mode to BCM standard
GPIO.setup(servoPin, GPIO.OUT) # set servoPin num to output
GPIO.setup(motorPin, GPIO.OUT) # set motorPin num to output
pwmServo = GPIO.PWM(servoPin, servoFreq) # initiate PWM with constant freq
pwmMotor = GPIO.PWM(motorPin, motorFreq)

#-----------Motor Receiver CALIBRATION
#for calibration: put DC (Duty Cycle) to max for 3 sec, to min for 2 sec, to nutral for 2 sec

pwmServo.start(servoStartDuty) # start servo/motor with their start Duty Cycle value (MAX)
pwmMotor.start(motorStartDuty)
time.sleep(3) 

pwmServo.ChangeDutyCycle(11.5) # change duty cycle to MIN
pwmMotor.ChangeDutyCycle(30)
time.sleep(2)

pwmServo.ChangeDutyCycle(14.5) # change duty cycle to nutral pos
pwmMotor.ChangeDutyCycle(19)
time.sleep(2)


def updateServo(angle): # updates servo by changing the DC
    dutyServo = float(angle) / 10.0 + 2.5
    pwmServo.ChangeDutyCycle(dutyServo)


def updateMotor(speed): # updates Motor speed by changing the DC
    dutyMotor = float(speed) / 10.0 + 15
    pwmMotor.ChangeDutyCycle(dutyMotor)

#---------------Main loop
angle = 120.0
speed = 40.0
while True:
    inputdata = open("bo", 'r') 
    
    angle_new = inputdata.readline() # read the 1st line for servo
    speed_new = inputdata.readline() # read the 2nd line for Motor
    
    angle_int = float(angle_new)
    speed_int = float(speed_new)
    
    
    if abs(angle - angle_int) > 0: # if the angle is the same, don't change 
        angle = angle_int
        updateServo(angle)
        print angle
        
    if abs(speed - speed_int) > 0: # if the speed is the same, don't change
        speed = speed_int
        updateMotor(speed)
        print speed
        
    inputdata.close
    time.sleep(0.5)
    

