//#include <esp_now.h>
//#include <WiFi.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ================== PINNIT ===================
#define STOP_BTN   18
#define BRAKE_BTN  19
#define JOY_X      32
#define JOY_Y      35
#define JOY_BTN    34
#define BAT_VOLT   33

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ================== RAKENNE LÄHETETTÄVILLE TIEDOILLE ===================
typedef struct {
  int L_PWM;
  int R_PWM;
  int L_DIR;
  int R_DIR;
  int BRAKE;
  int STOP;
} MotorData;

MotorData data;

// ================== APU ===================
int maxPWM = 100;
bool stopState = false;
bool brakeState = false;
unsigned long lastJoyBtn = 0;
unsigned long lastStopBtn = 0;
unsigned long lastBrakeBtn = 0;

// Vastaanottajan MAC-osoite (vaihda oman ESP32 vastaanottajan mukaan)
/*uint8_t receiverMac[] = {0x84, 0x1F, 0xE8, 0x1A, 0xAA, 0xD8};

void OnSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
  Serial.print("Lähetys: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "OK" : "FAIL");
}*/

// ================== ASETUS ===================
void setup() {
  Serial.begin(115200);

  pinMode(STOP_BTN, INPUT_PULLUP);
  pinMode(JOY_BTN, INPUT_PULLUP);
  pinMode(BRAKE_BTN,INPUT_PULLUP);
  // OLED init
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("SSD1306 init fail");
    for (;;);
  }
  display.clearDisplay();
  display.display();

  // ESP-NOW init
  /*WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init fail");
    while (true);
  }
  esp_now_register_send_cb(OnSent);
  esp_now_peer_info_t peerInfo;
  memcpy(peerInfo.peer_addr, receiverMac, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  peerInfo.ifidx = WIFI_IF_STA;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("ESP-NOW peer fail");
    while (true);
  }*/

  Serial.println("Kaukosäädin valmis");
}

// ================== LOOP ===================
void loop() {
  int xVal = analogRead(JOY_X) - 2048; // keskitys
  int yVal = analogRead(JOY_Y) - 2048;
  double srcV = (double)((float)analogRead(BAT_VOLT) * 0.00199f); //<-- 3.3 * (16.8/6.8) / 4056
  
    /*// Deadzone
  int deadzone = 400; // säädä tarvittaessa
  bool forward = (yVal > deadzone);
  bool backward = (yVal < -deadzone);
  bool right = (xVal > deadzone);
  bool left = (xVal < -deadzone);

  // Nollataan
  data.L_PWM = 0;
  data.R_PWM = 0;
  data.L_DIR = 0;
  data.R_DIR = 0;

  if (forward) {
    // Molemmat eteenpäin, mutta peilikuvina
    data.L_PWM = maxPWM;
    data.R_PWM = maxPWM;
    data.L_DIR = 1; // esim. vasen eteen
    data.R_DIR = 0; // oikea eteen (peilattu)
  } 
  else if (backward) {
    // Molemmat taaksepäin
    data.L_PWM = maxPWM;
    data.R_PWM = maxPWM;
    data.L_DIR = 0; 
    data.R_DIR = 1;
  } 
  else if (right) {
    // Paikallaan käännös oikealle
    data.L_PWM = maxPWM;
    data.R_PWM = maxPWM;
    data.L_DIR = 1; 
    data.R_DIR = 1;
  } 
  else if (left) {
    // Paikallaan käännös vasemmalle
    data.L_PWM = maxPWM;
    data.R_PWM = maxPWM;
    data.L_DIR = 0;
    data.R_DIR = 0;
  }*/

  // Joystickin nappi → maxPWM säätö
  if (!digitalRead(JOY_BTN) && millis() - lastJoyBtn > 300) {
    maxPWM += 10;
    if (maxPWM > 100) maxPWM = 10;
    lastJoyBtn = millis();
  }

  // STOP toggle
  if (!digitalRead(STOP_BTN) && millis() - lastStopBtn > 300) {
    stopState = !stopState;
    lastStopBtn = millis();
  }
  data.STOP = stopState ? 1 : 0;
  if (!digitalRead(BRAKE_BTN) && millis() - lastBrakeBtn > 300) {
    brakeState = !brakeState;
    lastBrakeBtn = millis();
  }
  data.BRAKE = brakeState ? 1 : 0;
  // Lähetys
  //esp_now_send(receiverMac, (uint8_t *) &data, sizeof(data));

  // OLED päivitys
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  display.printf("X:%d Y:%d\n",xVal, yVal);
  display.printf("STP:%d BRK:%d MAX:%d\n", data.STOP, data.BRAKE,maxPWM);
  display.printf("srcV: %f\n", srcV);
  display.display();

  delay(100);
}
