#include <Adafruit_NeoPixel.h>
#include <stdlib.h>

// Define the data pin and number of LEDs
#define DATA_PIN 3
#define NUM_LEDS 48

#define RING_START 0
#define RING_END 48

#define NO_MODE 0
#define SPARKLE 1

// Create NeoPixel object
Adafruit_NeoPixel strip = Adafruit_NeoPixel(NUM_LEDS, DATA_PIN, NEO_GRB + NEO_KHZ800);

unsigned long previousMillis = 0;
int mode = NO_MODE;

void setup() {

#ifdef RGB_BUILTIN
  digitalWrite(RGB_BUILTIN, HIGH);  // Turn the RGB LED white
  delay(20);
  digitalWrite(RGB_BUILTIN, LOW);  // Turn the RGB LED off
  delay(20);
  rgbLedWrite(RGB_BUILTIN, 255, 0, 0);  // Red
  delay(20);
  rgbLedWrite(RGB_BUILTIN, 0, 255, 0);  // Green
  delay(20);
  rgbLedWrite(RGB_BUILTIN, 0, 0, 255);  // Blue
  delay(20);
  rgbLedWrite(RGB_BUILTIN, 0, 0, 0);  // Off / black
  delay(20);
#endif

  strip.begin();  // Initialize the LED strip
  strip.show();   // Turn off all LEDs initially

  // startup_demo();
  fuzz(50);

  // int foo = 0;
  // while(true) {
  //   // foo = 4095
  //   foo = analogRead(9);
  //   rgbLedWrite(RGB_BUILTIN, foo / 256, 0, 0);
  // }

  // Start the serial communication
  Serial.begin(115200); // Ensure this matches the baud rate in the Python script

  while (!Serial) {


  //   fuzz(2);
    ; // Wait for serial port to connect. Needed for native USB.
    // demi_fuzz(); // add slow sparkle effect until USB is connected
  }

  // // startup_demo(); // serial now connected
  Serial.println("ESP32 is ready."); // Initial message for debugging

}

void set_pixel_range(int start,int end, int r, int g, int b) {
  for (int i = start; i <= end; i++) {
    strip.setPixelColor(i, strip.Color(r, g, b));
  }
  strip.show(); // Send data to the LEDs
}
void ring_white() {
  set_pixel_range(RING_START, RING_END, 255, 255, 255);
}
void ring_black() {
  set_pixel_range(RING_START, RING_END, 0, 0, 0);
}

void startup_demo() {
  int shift_size = 8;
  for (int i = 0; i < 50000; i++) {
    // // might be better to count up the colors instead or mod'ing them down
    // int foo = i;
    // int r = foo % 255;
    // foo = foo >> shift_size;
    // int g = foo % 255;
    // foo = foo >> shift_size;
    // int b = foo % 255;

    strip.setPixelColor((i-1) % NUM_LEDS, strip.Color(0, 0, 0));
    strip.setPixelColor(i % NUM_LEDS, strip.Color(20, 20, 20));
    strip.show(); // Send data to the LEDs
    // delay(20);
  }
}
void fuzz(int foo) {
  int str = 255;
  int idx = 0;
  
  for (int i = 0; i < foo; i++) {
    strip.setPixelColor(idx, strip.Color(0, 0, 0));
    idx = rand() % NUM_LEDS;
    // strip.setPixelColor(idx, strip.Color(str, str, str));
    strip.setPixelColor(idx, strip.Color(rand() % 255, rand() % 255, rand() % 255));
    strip.show(); // Send data to the LEDs

    delay(rand() % 100); // good for keeping a slower sparkle
  }


    strip.setPixelColor(idx, strip.Color(0, 0, 0));
    strip.show(); // Send data to the LEDs
}

// void set_all_pins

int demi_fuzz_i = 0;
void demi_fuzz() {
    strip.setPixelColor(demi_fuzz_i, strip.Color(0, 0, 0));
    demi_fuzz_i = rand() % NUM_LEDS;
    strip.setPixelColor(demi_fuzz_i, strip.Color(rand() % 255, rand() % 255, rand() % 255));
    strip.show();
    // delay(rand() % 100);
}


void demi_sparkle() {

}
// int old = 0;
int pixel_until = 0;
void loop() {

  // if 


  // while (true) {
    // rgbLedWrite(RGB_BUILTIN, 0, 0, 0);
    // Reading potentiometer value
    // int val = analogRead(4);
    // Serial.println(val - 3200);
    // Serial.println(42);
    // delay(500);
  // }

  // strip.setPixelColor(old, strip.Color(0, 0, 0));
  // old = val / 255;
  // strip.setPixelColor(old, strip.Color(val%255, 0, 0));
  // strip.show();


  // Check if data is available to read
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n'); // Read the incoming command
    command.trim(); // Remove leading/trailing whitespace

    // TODO: make so that commands make it stop doing whatever it's at at the moment.


    // Respond to specific commands
    if (command == "ping") {
      Serial.println("pong"); // Respond to "ping" command
    } else if (command == "status") {
    } else if (command == "white") {
      ring_white();
      mode = NO_MODE;
    } else if (command == "black") {
      ring_black(); 

      mode = SPARKLE;
    } else if (command == "sparkle") {
      mode = SPARKLE;
    } else if (command == "bar") {
      fuzz(50);
    } else if (command == "biz") {
      fuzz(250);
      Serial.println("ESP32 is running fine."); // Respond to "status" command
    } else {
      Serial.println("Unknown command: " + command); // Handle unknown commands
    }
  }

  if (mode == SPARKLE) {
    // background dazzle, maybe make it sleep for a while after/during activity
    unsigned long currentMillis = millis();
    // check if it's beyong pixel_until
    if (pixel_until < currentMillis) {
      pixel_until = currentMillis + rand()%500;
      demi_fuzz();  // Call your ticking function
    }
    // //
  }

//  Check if one second has passed
  // if (currentMillis - previousMillis >= 1000) {
  //   previousMillis = currentMillis;
  //   demi_fuzz();  // Call your ticking function
  // }


}