

//basic functions



//games

void talletus(){
  lcd.print("Talletetaan");
  delay(gameSpeed);
  lcdBottomClear();
  int deposit = drinkCount();
  lcd.print(deposit);
  lcd.print(" huikkaa");
  bank += deposit;
  
}

void juo(){
  lcd.print(drinkCount());
  lcd.print(" huikkaa");
  
}

void shotgun(){
    bool win = coinToss();
    //Serial.println(win);
    if(!win){
      fill_solid(leds, NUM_LEDS, CRGB::Red);
      lcd.print("SHOTGUN");
    } else{
      fill_solid(leds, NUM_LEDS, CRGB::Green);
      lcd.print("GG");
    }
    FastLED.show();
    gameDelay(gameSpeed * 5);
    totalSips += 30;
}


void pankki(){
  int splits = playersToDrink();
  lcd.print("Jaetaan ");
  lcd.print(splits);
  lcd.print(" osaan");
  delay(gameSpeed);
  lcdBottomClear();
  lcd.print(bank / splits);
  lcd.print(" per turpa");
  totalSips += bank;
  bank = 0;
  gameDelay(gameSpeed * 5);
}
