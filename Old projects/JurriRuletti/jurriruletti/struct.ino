

void start(){
 
 lcd.print("pelaajia: ");
 lcd.setCursor(0,1);
 while(!button2pressed || playerCount == 0){
  button1.tick();
  button2.tick();
  
  lcd.setCursor(0,1);
  lcd.print(playerCount);
  if(button1pressed){
    playerCount = (playerCount + 1) % 9;
    fill_solid(leds, NUM_LEDS, CRGB::Black);
    for(int a = 0; a < playerCount*2; a +=2){
    leds[a] = CRGB::Blue;
    }
    FastLED.show();
    button1pressed = false;
  }
  if(button2pressed && playerCount == 0){
    button2pressed = false;
  }
 }
 button2pressed = false;
 
 lcd.clear();
 lcd.home();
 lcd.print("vaikeustaso: ");
 
 for(int a = 16; a  < NUM_LEDS; a++){
    leds[a] = CHSV(96 - difficulity*48,255,255);
    FastLED.show();
  }
 while(!button2pressed){
  button1.tick();
  button2.tick();
  
  lcd.setCursor(0,1);
  lcd.print(difficulities[difficulity]);
  
  if(button1pressed){
    difficulity = (difficulity + 1) % 3;
    for(int a = 16; a  < NUM_LEDS; a++){
      leds[a] = CHSV(96 - difficulity*48,255,255);
      FastLED.show();
    }
    lcd.setCursor(0,1);
    lcd.print("    ");
    button1pressed = false;
  }
  
 }
 button2pressed = false;
 maxGameRounds = 3 + difficulity * 3;
 gameTime = millis();
}

void ending(){
  if(bank > 0){
    pankki();
  }
  double playTimeMin = (millis() -  gameTime)  / 60000.0;
  lcd.clear();
  lcd.setCursor(4,0);
  lcd.print("LOPPU");
  
  while(1){
    lcdBottomClear();
    lcd.print("Huikkia: ");
    lcd.print(totalSips);
    delay(gameSpeed);
    lcdBottomClear();
    lcd.print("Aika: ");
    lcd.print(playTimeMin);
    lcd.print(" min");
    delay(gameSpeed);
  }
  
}

int roulette(){
  int wheelStop = false;
  int randomSpot =  weightedRandom();
  int delayVal = 50;
  while(!wheelStop){
    
    //leds[ball] = CRGB::Black;
    fadeToBlackBy(leds, NUM_LEDS, 128); // fades 25% to black, all leds
    ball = (ball + 1) % 16;
    leds[ball] = CRGB::Red;
    FastLED.show();
    
    noTone(piezoPin);
    tone(piezoPin, 100 + delayVal,1000);
    delay(delayVal);
    
    delayVal += delayStep;
    
    if(delayVal > 200){
      if(ball == randomSpot){
        wheelStop = true;
      }
    }
    
  }
  fill_solid(leds, NUM_LEDS, CRGB::Black);
  leds[ball] = CRGB::Red;
  FastLED.show();
  return ball;
}

bool coinToss(){
  int spinning = true;
  int anim = 0;
  int randomSpot = random(0,15);
  
  int delayVal = 50;
  leds[16] = CRGB::Blue;
  while(spinning){
    anim = (anim + 1) % 16;
    for (int i = 0; i < 16; i++) {
      if ((i + anim) % 16 < 12 - difficulity * 4) { // "12 - difficulity * 4" is that difficulity 0(ez) = 75% win, 1(norm)= 50%, and hc(25%)
        leds[i] = CRGB::Red; // Ensimmäinen puolisko vihreä
      } else {
        leds[i] = CRGB::Green;   // Toinen puolisko punainen
        }
    }
    delay(delayVal);
    FastLED.show();
    if((anim % 2) == 0){
      tone(piezoPin, 200,500);
    }else{
      tone(piezoPin, 300,500);
    }
    delayVal +=(delayStep);
    if(delayVal > 300){
      spinning = anim != randomSpot; 
    }
  }
  return leds[0] == CRGB::Red; // returns 1 = win, 0 = fail
  
}

//dev tools for testing
//uses forced and forcedVal to change the outcomes or straight manipulates global variables
void devTools(){
   
  lcd.clear();
  lcd.home();
  lcd.print("devTools");
  lcd.setCursor(0,1);
  
  int outp = inputs();// outp is values SVVV... S = select 1 - 9, V = values 100
  if(outp == 0) return;
  lcdBottomClear();
  
  int v = extractValue(outp);
  int s = extractFunction(outp);
  
  char buff[16];
  sprintf(buff, "Verify: %d-%d", s, v); // show the current command: Verify: S-Val
  lcd.print(buff);
  while(!button2pressed){ // works as verification, red button - > yes, other - > no
    button1.tick();
    button2.tick();
    if(button1pressed){
      button1pressed = false;
      return;
    }
  }
  
  button2pressed = false;
  
  switch(s){
    case 1:
      if(v >= 0 && v < 16){
        forced = true;
        forcedVal = v;
      }
      break;

    case 2:
      bank = v;
      break;
      
    case 3:
      difficulity = v;
      break;
      
    case 4:
      multiplier = v;
      break;
      
    case 5:
      playerCount = v;
      break;
      
    case 6:
      gameTurn = v;
      break;
      
    case 7:
      gameRound = v;
      break;
      
    case 8:
      maxGameRounds = v;
      break;
      
    case 9:
      ending();
      break;
  }
  tone(piezoPin, 1000, 500);
}
