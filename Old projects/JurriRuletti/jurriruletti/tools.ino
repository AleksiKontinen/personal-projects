




// using the 2 buttons you can input a number
int inputs(){
  long timeNow = millis();
  int presses = 0;
  bool refresh = true;
  while(millis() - timeNow < 3000){
    button1.tick();
    button2.tick();
    if(button1pressed){
      button1pressed = false;
      timeNow = millis();
      presses++;
      refresh = true;
    }
    if(button2pressed){
      button2pressed = false;
      timeNow = millis();
      presses += 10;
      refresh = true;
    }

    if(refresh){
      lcdBottomClear();
      lcd.print(presses);
      refresh = false;
    }
    
  }
  return presses;
}

void lcdShowDefault(){
  // P A N K K I : X X X    K  E  :  X
  // V : X / X   K : X / X     
  // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
  lcd.home();
  lcd.clear();
  char buff1[16];
  sprintf(buff1,"Pankki:%d Ke:%d",bank,multiplier);
  lcd.print(buff1);
  lcd.setCursor(0,1);
  char buff2[16];
  sprintf(buff2,"V:%d/%d K:%d/%d",gameTurn,playerCount,gameRound,maxGameRounds);
  lcd.print(buff2);
}

//basicly just a delay, but skippable.
void gameDelay(int delayLen){
  long timeNow = millis();
  while(millis() - timeNow < delayLen){
    button2.tick();
    button1.tick();
    if(button1pressed || button2pressed){
      button1pressed = false;
      button2pressed = false;
      break;
    }
  }
}


int playersToDrink(){
  int drinking;
  if(playerCount > 2){
    drinking = random(1,playerCount - difficulity);
  } else {
    drinking = random(1,playerCount);
  }
  
  //Serial.println(drinking);
  for(int a = 0; a  < drinking; a++){
    leds[random(0,15)] = CRGB :: Red;
    FastLED.show();
    tone(piezoPin, 200,500);
    delay(1000); 
    }

  return drinking;
}

int drinkCount(){
  int drinks = random(difficulity*2 , 3*difficulity + 3) * multiplier; 
  //Serial.println(drinks);
  for(int a = 0; a < drinks; a++){
    if(a > 11){
      leds[a+4] = CRGB :: Purple;
    }else{
      leds[a+16] = CRGB :: Red;
    }
    noTone(piezoPin);
    tone(piezoPin, 50 + 10*a,200);
    FastLED.show();
    delay(100);
  }
  delay(gameSpeed);
  for(int a = 16; a  < NUM_LEDS; a++){
    leds[a] = CRGB::Black;
    FastLED.show();
  }
  totalSips += drinks;
  return drinks;
}

int weightedRandom() {
  if(forced){
    forced = false;
    return forcedVal;
  }
  float weights[16];
  float totalWeight = 0;

  // Lasketaan jokaiselle numerolle painotus: mitä vähemmän tullut, sitä suurempi paino
  for (int i = 0; i < 16; i++) {
    weights[i] = 1.0 / (counts[i] + 1);  // Pienet luvut saavat suuremman todennäköisyyden
    totalWeight += weights[i];
  }

  // Arvotaan luku painotetusti
  float r = (float)random(10000) / 10000.0 * totalWeight;  // Satunnaisluku välillä [0, totalWeight]
  float sum = 0;
  for (int i = 0; i < 16; i++) {
    sum += weights[i];
    if (r <= sum) {
      counts[i]++;  // Kasvatetaan valitun numeron laskuria
      return i;
    }
  }

  return 0;  // Pitäisi olla mahdotonta päästä tänne
}

//All button functions

void click1(){
  button1pressed = true;
  tone(piezoPin, 200, 100);
}
void click2() {
   button2pressed = true;
   tone(piezoPin, 400, 100);
}
void doubleClick1(){
  doubleClick = true;
  tone(piezoPin,600,100);
}

void longPressStart2(){
  longPress = true;
}
void longPressStop2(){
  longPress = false;
}

void lcdBottomClear(){ //clears bottom part leaving cursor at bottom row first spot
  lcd.setCursor(0,1);
  lcd.print("                ");
  lcd.setCursor(0,1);
}

int extractFunction(int num) {
    while (num >= 10) {  // Poistaa kaikki paitsi ensimmäisen numeron
        num /= 10;
    }
    return num;
}

int extractValue(int num) {
    int factor = 1;
    while (num >= 10) {  // Selvitetään jakaja, esim. 9110 -> 1000
        factor *= 10;
        num /= 10;
    }
    return num % factor;  // Poistetaan ensimmäinen numero
}
