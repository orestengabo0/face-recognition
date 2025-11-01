// ============================================
// Motor Control for Face Tracking (Fast Version)
// ============================================

// Motor pins connected to ULN2003 driver
const int motorPin1 = 8;   // IN1
const int motorPin2 = 9;   // IN2
const int motorPin3 = 10;  // IN3
const int motorPin4 = 11;  // IN4

// Motor specifications
const int stepsPerRevolution = 2048;  // 28BYJ-48 motor
int stepDelay = 1;  // Faster speed (1 ms between steps)

// Step sequence for 28BYJ-48 (half-step, smooth motion)
const int stepSequence[8][4] = {
  {1, 0, 0, 1},
  {1, 0, 0, 0},
  {1, 1, 0, 0},
  {0, 1, 0, 0},
  {0, 1, 1, 0},
  {0, 0, 1, 0},
  {0, 0, 1, 1},
  {0, 0, 0, 1}
};

// Current step index
int currentStep = 0;

// ============================================
// SETUP
// ============================================
void setup() {
  Serial.begin(9600);

  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  pinMode(motorPin3, OUTPUT);
  pinMode(motorPin4, OUTPUT);

  stopMotor();

  Serial.println("Arduino Ready - Face Tracking Motor Controller (Fast Mode)");
  Serial.println("Waiting for commands...");
}

// ============================================
// MAIN LOOP
// ============================================
void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.length() > 0) {
      processCommand(command);
    }
  }
}

// ============================================
// COMMAND PROCESSING
// ============================================
void processCommand(String cmd) {
  if (cmd == "S") {
    stopMotor();
    Serial.println("Motor stopped");
  }
  else if (cmd.startsWith("L:")) {
    int steps = cmd.substring(2).toInt();
    if (steps > 0) {
      moveMotor(steps, -1);
      Serial.print("Moved LEFT ");
      Serial.print(steps);
      Serial.println(" steps");
    }
  }
  else if (cmd.startsWith("R:")) {
    int steps = cmd.substring(2).toInt();
    if (steps > 0) {
      moveMotor(steps, 1);
      Serial.print("Moved RIGHT ");
      Serial.print(steps);
      Serial.println(" steps");
    }
  }
  else if (cmd.startsWith("SPD:")) {  // optional speed change support
    int newDelay = cmd.substring(4).toInt();
    setSpeed(newDelay);
  }
  else {
    Serial.print("Unknown command: ");
    Serial.println(cmd);
  }
}

// ============================================
// MOTOR CONTROL FUNCTIONS
// ============================================
void moveMotor(int steps, int direction) {
  for (int i = 0; i < steps; i++) {
    digitalWrite(motorPin1, stepSequence[currentStep][0]);
    digitalWrite(motorPin2, stepSequence[currentStep][1]);
    digitalWrite(motorPin3, stepSequence[currentStep][2]);
    digitalWrite(motorPin4, stepSequence[currentStep][3]);

    currentStep += direction;
    if (currentStep >= 8) currentStep = 0;
    else if (currentStep < 0) currentStep = 7;

    delay(stepDelay);
  }
}

void stopMotor() {
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, LOW);
  digitalWrite(motorPin3, LOW);
  digitalWrite(motorPin4, LOW);
}

void setSpeed(int delayMs) {
  stepDelay = constrain(delayMs, 1, 10);
  Serial.print("Speed updated: stepDelay = ");
  Serial.println(stepDelay);
}
