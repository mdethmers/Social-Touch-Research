// Define servo pins
int servoPins[] = {26, 27, 25};
#define NUM_SERVOS (sizeof(servoPins) / sizeof(servoPins[0]))

// Motor driver pins (adapted from your working code)
#define MOTOR_IN1 2 // Driver 1 - Pump 1 PWM pin
#define MOTOR_IN2 15 // Driver 1 - Pump 1 direction pin

#define MOTOR_IN3 12 // Driver 2 - Pump 2 PWM pin  
#define MOTOR_IN4 13 // Driver 2 - Pump 2 direction pin

// Servo state tracking
int currentPositions[NUM_SERVOS];
unsigned long lastPulseTime[NUM_SERVOS];
unsigned long lastSerialCheck = 0;

// Motor state
int currentMotorValue = 0;

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("Servo + Motor control starting...");
    
    // Set up servo pins
    for (int i = 0; i < NUM_SERVOS; i++) {
        pinMode(servoPins[i], OUTPUT);
        lastPulseTime[i] = 0;
        Serial.printf("Servo pin %d configured\n", servoPins[i]);
        delay(10);
    }
    
    // Set up motor driver pins
    pinMode(MOTOR_IN1, OUTPUT);
    pinMode(MOTOR_IN2, OUTPUT);
    pinMode(MOTOR_IN3, OUTPUT);
    pinMode(MOTOR_IN4, OUTPUT);
    
    // Initialize motors to off (using your method)
    stopPump1();
    stopPump2();
    
    Serial.println("Setup complete - ready for commands");
    Serial.println("Send commands as: index,value");
    Serial.println("Index 3 controls air pumps: +255 = pump 1, -255 = pump 2, 0 = off");
}

void loop() {
    unsigned long currentTime = micros();
    
    // Update servo pulses (non-blocking)
    updateServoPositions(currentTime);
    
    // Check for serial data every 10ms
    if (currentTime - lastSerialCheck >= 50000) {
        checkSerial();
        lastSerialCheck = currentTime;
    }
}

void updateServoPositions(unsigned long currentTime) {
    for (int i = 0; i < NUM_SERVOS; i++) {
        // Send pulse every 20ms (20000 microseconds)
        if (currentTime - lastPulseTime[i] >= 20000) {
            digitalWrite(servoPins[i], HIGH);
            delayMicroseconds(currentPositions[i]);
            digitalWrite(servoPins[i], LOW);
            
            lastPulseTime[i] = currentTime;
        }
    }
}

void checkSerial() {
    static String inputBuffer = "";
    
    while (Serial.available()) {
        char incomingChar = Serial.read();
        
        if (incomingChar == '\n' || incomingChar == '\r') {
            if (inputBuffer.length() > 0) {
                processSerialCommand(inputBuffer);
                inputBuffer = "";
            }
        } else {
            inputBuffer += incomingChar;
            
            if (inputBuffer.length() > 50) { //gooit alle waarde weg, wat dus wellicht de hoogste waarde is, omdat de buffer te lang is. 
                Serial.println("Input buffer overflow - clearing");
                inputBuffer = "";
            }
        }
    }
}

void processSerialCommand(String command) {
    command.trim();
    Serial.println("Received: " + command);
    
    int commaIndex = command.indexOf(',');
    if (commaIndex > 0 && commaIndex < command.length() - 1) {
        int index = command.substring(0, commaIndex).toInt();
        int value = command.substring(commaIndex + 1).toInt();
        
        if (index == 3) {
            // Handle motor control (air pumps)
            controlPumps(value);
        } else if (index >= 0 && index < NUM_SERVOS) {
            // Handle servo control
            value = constrain(value, 500, 2500);
            currentPositions[index] = value;
            Serial.printf("Servo %d set to %d microseconds\n", index, value);
        } else {
            Serial.printf("Invalid index: %d\n", index);
        }
    } else {
        Serial.println("Invalid format. Use: index,value");
    }
}

void controlPumps(int value) {
    currentMotorValue = value;
    
    if (value > 0) {
        stopPump2();
        startPump1(value); 
        Serial.printf("Pump 1 ON (value: %d)\n", value);
        
    } else if (value < 0) {
        stopPump1();
        startPump2(abs(value)); 
        Serial.printf("Pump 2 ON (value: %d)\n", abs(value));
        
    } else {
        // Zero: turn off both pumps
        stopPump1();
        stopPump2();
        Serial.println("All pumps OFF");
    }
}

void startPump1(int speed) {
    digitalWrite(MOTOR_IN2, LOW);
    analogWrite(MOTOR_IN1, speed);
    
    analogWrite(MOTOR_IN3, 0); 
    digitalWrite(MOTOR_IN4, LOW);
}

void startPump2(int speed) {
    digitalWrite(MOTOR_IN4, LOW);
    analogWrite(MOTOR_IN3, speed);
    
    analogWrite(MOTOR_IN1, 0);  
    digitalWrite(MOTOR_IN2, LOW);
}

void stopPump1() {
    analogWrite(MOTOR_IN1, 0);  
    digitalWrite(MOTOR_IN2, LOW);
}

void stopPump2() {
    analogWrite(MOTOR_IN3, 0);  
    digitalWrite(MOTOR_IN4, LOW);
}