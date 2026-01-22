
//Roulette type drinking game weekend project
//Consists of arduino nano, 2 different sized programmable led rings, lcd display and few buttons.
//Housing made from some random box


#include <OneButton.h>
#include <LiquidCrystal_I2C.h>
#include <FastLED.h>


//Kytkännät
#define NUM_LEDS 28
#define PIN_INPUT1 A1
#define PIN_INPUT2 A2
#define DATA_PIN 6
#define piezoPin 9

//Softa
#define ledBrightness 20
#define delayStep 5
#define gameSpeed 2000
//kirjastot
LiquidCrystal_I2C lcd(0x27, 16, 2); 
CRGB leds[NUM_LEDS];
OneButton button1(PIN_INPUT1, true);
OneButton button2(PIN_INPUT2, true);

//globaalit muuttujat
unsigned long gameTime = 0;
unsigned int bank = 0,ball = 0,playerCount = 0,difficulity = 1, multiplier = 1, gameRound = 1, gameTurn = 1,maxGameRounds = 3,totalSips = 0, forcedVal;
const String spotNames[] = {"talletus", "juo", "totuus", "missio", "laki", "otto", "kerroin", "villikortti", "shot(gun)", "peli"};
uint8_t counts[16];
const String difficulities[] = {"ez","norm","hc"};

const int rouletteSpots[16] = {5,0,2,1,4,6,0,7,9,0,3,1,8,9,6,0}; // kohdat johon pelit asettuvat 0-15 0,0,0,0,1,1,2,3,4,5,6,6,7,8,9,9
bool button1pressed = false, button2pressed = false, forced = false,longPress = false, doubleClick = false; 

                            
void setup() {
  // put your setup code here, to run once:

  lcd.init(); // initialize the //lcd
  lcd.backlight();
  lcd.clear();
  lcd.home();
   FastLED.addLeds<NEOPIXEL, DATA_PIN>(leds, NUM_LEDS);
   FastLED.addLeds<WS2812, DATA_PIN, GRB>(leds, NUM_LEDS);
   FastLED.setBrightness(ledBrightness);
   
   button1.attachClick(click1); // smaller settign button
   button1.attachDoubleClick(doubleClick1);
   
   button2.attachClick(click2); //big red Button
   button2.attachLongPressStart(longPressStart2);
   button2.attachLongPressStop(longPressStop2);
   
   randomSeed(analogRead(0));
   // init led Brightness
   for(int a = 0; a < 16; a++){
    counts[a] = 0;
   }
   //Serial.begin(9600);
   
   start();
}

void loop() { //main funktion
 
  // put your main code here, to run repeatedly:
  fill_solid(leds, NUM_LEDS, CRGB::Black);
  FastLED.show();
  delay(100);
  for(int a = 0; a < 16; a++){
    leds[a] = CHSV(255 - rouletteSpots[a]*25, 255 ,255);
    delay(10);
  }
  
  leds[ball] = CRGB::White;
  FastLED.show();
  
  lcdShowDefault();
  
  while(!button2pressed){
    button2.tick();
    button1.tick();
    if(longPress && doubleClick){
      devTools();
      break;
    }
    delay(10);
  }
  button2pressed = false;
  doubleClick = false;
  int game = rouletteSpots[roulette()];
  
  
  //Serial.print(game);
  //Serial.print(spotNames[game]);
  
  
  lcd.clear();
  lcd.home();
  lcd.print(spotNames[game]);
  gameDelay(gameSpeed);
  lcdBottomClear();
  fill_solid(leds, NUM_LEDS, CRGB::Black);
  FastLED.show();
  //valinta struc
  switch(game){
    case 0:
      talletus();
      break;
      
    case 1:
      juo();
      break;
      
    case 2:
      //totuus
      break;
      
    case 3:
      //tehtävä
      break;
      
    case 4:
      //sääntö
      break;
      
    case 5:
      pankki();
      break;
      
    case 6:
      multiplier = random(1,3 + difficulity);
      lcd.print(multiplier);
      break;
      
    case 7:
      //villikortti
      break;
    case 8:
      shotgun();
      break;

    case 9:
       // peli
      break;
     
  }
  gameDelay(gameSpeed);
  gameTurn++;
  bank *= (1 + 0.05*difficulity); // korko 0% - 10% / per kierros (ez-hc)
  if(gameTurn > playerCount){
    gameTurn = 1;
    gameRound++;
  }
  if(gameRound > maxGameRounds){
    lcd.clear();
    lcdBottomClear();
    ending();
    
  }
}




  
