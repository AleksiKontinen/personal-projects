/*
  Ohjelma 28BYJ-48 askelmoottorin ohjaamiseksi ULN2003-ajurilla.
  Nopeus asetetaan Sarjamonitorin kautta (-100 - 100).

  Vaatii "AccelStepper" -kirjaston.
*/

#include <AccelStepper.h>

// Määritä moottorin liitännät.
// Kytke nämä Arduino-pinnit ULN2003-piirin pinneihin IN1, IN2, IN3, IN4.
// Järjestys (1, 3, 2, 4) on yleensä oikea tälle moottorille half-step-tilassa.
#define MOTOR_PIN_1 8
#define MOTOR_PIN_2 9
#define MOTOR_PIN_3 10
#define MOTOR_PIN_4 11

// Käytämme 4-johdinta ja "half-step" -tilaa (HALF4WIRE), joka on tasaisin.
// Pinnijärjestys AccelStepper-kirjastolle: (pin1, pin3, pin2, pin4)
AccelStepper stepper(AccelStepper::HALF4WIRE, MOTOR_PIN_1, MOTOR_PIN_3, MOTOR_PIN_2, MOTOR_PIN_4);

// Määritetään suurin nopeus askelta sekunnissa.
// 28BYJ-48 -moottorille n. 1000 steps/sec on hyvä maksimi.
// Tämä vastaa käyttäjän antamaa arvoa 100.
const int MAX_SPEED_STEPS = 1000;

void setup() {
  Serial.begin(9600); // Käynnistä sarjayhteys
  Serial.println("Askelmoottorin ohjaus");
  Serial.println("Anna nopeus väliltä -100 (taakse) - 100 (eteen):");

  // Aseta moottorin maksiminopeus ja pieni kiihtyvyys pehmeää käynnistystä varten
  stepper.setMaxSpeed(MAX_SPEED_STEPS);
  stepper.setAcceleration(500); // askelta / sekunti^2
}

void loop() {
  // 1. Tarkista, onko sarjaportista tullut dataa
  if (Serial.available() > 1) {
    // Lue numero sarjaportista
    int speedPercent = Serial.parseInt();

    // Rajoita arvo välille -100 ja 100
    speedPercent = constrain(speedPercent, -100, 100);

    // Muunna prosenttiarvo (-100...100)
    // moottorin todelliseksi nopeudeksi (-MAX_SPEED...MAX_SPEED)
    long motorSpeed = map(speedPercent, -100, 100, -MAX_SPEED_STEPS, MAX_SPEED_STEPS);

    // Aseta moottorille uusi haluttu nopeus
    stepper.setSpeed(motorSpeed);

    // Tulosta vahvistus
    Serial.print("Asetetaan nopeus: ");
    Serial.print(speedPercent);
    Serial.print("% (");
    Serial.print(motorSpeed);
    Serial.println(" askelta/s)");

    // Tyhjennä loput puskurista (kuten rivinvaihto)
    while (Serial.available() > 0) {
      Serial.read();
    }
  }

  // 2. Aja moottoria
  // Tätä funktiota täytyy kutsua niin usein kuin mahdollista.
  // Se tarkistaa, onko aika ottaa askel asetetulla nopeudella.
  // Tämä EI pysäytä koodin suoritusta (ei-blokkaava).
  stepper.runSpeed();
}