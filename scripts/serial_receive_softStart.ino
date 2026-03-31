// ---------------- Servo Setup ----------------
int servoPins[] = {26, 27, 25};
#define NUM_SERVOS (sizeof(servoPins) / sizeof(servoPins[0]))

int currentPositions[NUM_SERVOS];
unsigned long lastPulseTime[NUM_SERVOS];

// ---------------- Relay Setup ----------------
#define RELAY1_PIN 13   // Relay 1 signal
#define RELAY2_PIN 15   // Relay 2 signal

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("Servo + Relay control ready");

    // Initialize servos
    for (int i = 0; i < NUM_SERVOS; i++) {
        pinMode(servoPins[i], OUTPUT);
        currentPositions[i] = 1500;  // Start at center position
        lastPulseTime[i] = 0;
    }

    // Initialize relays
    pinMode(RELAY1_PIN, OUTPUT);
    pinMode(RELAY2_PIN, OUTPUT);
    digitalWrite(RELAY1_PIN, LOW);
    digitalWrite(RELAY2_PIN, LOW);

    Serial.println("Commands: index,value (0-2=servos, 3=relays)");
}

void loop() {
    unsigned long currentTime = micros();
    
    // Check for serial commands every loop
    checkSerial();
    
    // Send servo pulses
    updateServoPositions(currentTime);
}

void updateServoPositions(unsigned long currentTime) {
    for (int i = 0; i < NUM_SERVOS; i++) {
        // Send pulse every 20ms
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
        char c = Serial.read();
        
        if (c == '\n' || c == '\r') {
            if (inputBuffer.length() > 0) {
                processSerialCommand(inputBuffer);
                inputBuffer = "";
            }
        } else if (inputBuffer.length() < 50) {
            inputBuffer += c;
        } else {
            // Buffer overflow - clear
            inputBuffer = "";
        }
    }
}

void processSerialCommand(String command) {
    command.trim();
    
    int commaIndex = command.indexOf(',');
    if (commaIndex > 0 && commaIndex < command.length() - 1) {
        int index = command.substring(0, commaIndex).toInt();
        int value = command.substring(commaIndex + 1).toInt();

        if (index == 3) {
            // Control relays
            controlRelays(value);
        } else if (index >= 0 && index < NUM_SERVOS) {
            // Control servo - direct update
            value = constrain(value, 500, 2500);
            currentPositions[index] = value;
            Serial.printf("Servo %d -> %d µs\n", index, value);
        } else {
            Serial.printf("Invalid index: %d\n", index);
        }
    } else {
        Serial.println("Invalid format");
    }
}

void controlRelays(int value) {
    // Relay 1 ON if value > 127
    digitalWrite(RELAY1_PIN, (value > 127) ? HIGH : LOW);
    
    // Relay 2 ON if value < -127
    digitalWrite(RELAY2_PIN, (value < -127) ? HIGH : LOW);
    
    Serial.printf("Relays: R1=%s R2=%s (value=%d)\n",
                  digitalRead(RELAY1_PIN) ? "ON" : "OFF",
                  digitalRead(RELAY2_PIN) ? "ON" : "OFF",
                  value);
}