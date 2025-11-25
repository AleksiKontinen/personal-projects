// Pinimääritykset
#define STOP_PIN   16  // aktiivinen high
#define LDIR_PIN   17
#define RDIR_PIN    5
#define BRAKE_PIN  18  // aktiivinen high
#define LPWM_PIN   19
#define RPWM_PIN   21

// Tilamuuttujat
int Lpwm = 0;
int Rpwm = 0;
bool Ldir = 0;
bool Rdir = 0;
bool StopState = 0;
bool BrakeState = 0;


#define PWM_FREQ 20000


String inputString = "";

void printState() {
  Serial.print("L: "); Serial.print(Lpwm);
  Serial.print(" d: "); Serial.print(Ldir);
  Serial.print("; R: "); Serial.print(Rpwm);
  Serial.print(" d: "); Serial.print(Rdir);
  Serial.print("; B: "); Serial.print(BrakeState);
  Serial.print("; S: "); Serial.println(StopState);
}

void setup() {
  Serial.begin(115200);

  pinMode(STOP_PIN, OUTPUT);
  pinMode(LDIR_PIN, OUTPUT);
  pinMode(RDIR_PIN, OUTPUT);
  pinMode(BRAKE_PIN, OUTPUT);

  // PWM asetukset
  
  ledcAttach(LPWM_PIN, PWM_FREQ, 8);
  ledcAttach(RPWM_PIN, PWM_FREQ, 8);

  // Alustustilat
  digitalWrite(STOP_PIN, LOW);
  digitalWrite(LDIR_PIN, Ldir);
  digitalWrite(RDIR_PIN, Rdir);
  digitalWrite(BRAKE_PIN, LOW);
  ledcWrite(LPWM_PIN, Lpwm);
  ledcWrite(RPWM_PIN, Rpwm);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputString.length() > 0) {
        processCommand(inputString);
        inputString = "";
      }
    } else {
      inputString += c;
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();

  if (cmd.startsWith("L") && cmd.length() > 1 && isDigit(cmd[1])) {
    Lpwm = cmd.substring(1).toInt();
    Lpwm = constrain(Lpwm, 0, 255);
    ledcWrite(LPWM_PIN, Lpwm);
  }
  else if (cmd.startsWith("R") && cmd.length() > 1 && isDigit(cmd[1])) {
    Rpwm = cmd.substring(1).toInt();
    Rpwm = constrain(Rpwm, 0, 255);
    ledcWrite(RPWM_PIN, Rpwm);
  }
  else if (cmd == "LD") {
    Ldir = !Ldir;
    digitalWrite(LDIR_PIN, Ldir);
  }
  else if (cmd == "RD") {
    Rdir = !Rdir;
    digitalWrite(RDIR_PIN, Rdir);
  }
  else if (cmd == "S") {
    StopState = !StopState;
    digitalWrite(STOP_PIN, StopState);
  }
  else if (cmd == "B") {
    BrakeState = !BrakeState;
    digitalWrite(BRAKE_PIN, BrakeState);
  }

  printState();
}