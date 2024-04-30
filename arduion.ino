// Define the LED cube dimensions
#define CUBE_SIZE_X 8
#define CUBE_SIZE_Y 8
#define CUBE_SIZE_Z 8

// Define eye and mouth patterns
bool eyesPattern[CUBE_SIZE_Y][CUBE_SIZE_X] = {
  {false, false, false, false, true, true, false, false},
  {false, false, false, false, true, true, false, false},
  {false, false, false, false, false, false, false, false},
  {false, false, false, false, false, false, false, false},
  {false, false, false, false, false, false, false, false},
  {false, false, false, false, false, false, false, false},
  {false, false, false, false, false, false, false, false},
  {false, false, false, false, false, false, false, false}
};

bool mouthPattern[CUBE_SIZE_Y][CUBE_SIZE_X] = {
  {false, false, false, false, false, false, false, false},
  {false, true, true, true, true, true, true, false},
  {false, true, false, false, false, false, true, false},
  {false, true, false, false, false, false, true, false},
  {false, true, false, false, false, false, true, false},
  {false, true, false, false, false, false, true, false},
  {false, true, false, false, false, false, true, false},
  {false, false, false, false, false, false, false, false}
};

// Function to render the cube
void renderCube() {
  for (int z = 0; z < CUBE_SIZE_Z; z++) {
    digitalWrite(SS, LOW);
    SPI.transfer(0x01 << z);
    for (int y = 0; y < CUBE_SIZE_Y; y++) {
      byte row = 0;
      for (int x = 0; x < CUBE_SIZE_X; x++) {
        // Set the appropriate bit in the row byte if LED is on
        if ((z == 3 || z == 4) && eyesPattern[y][x]) // Adjust the Z-axis for eyes
          row |= (1 << (CUBE_SIZE_X - 1 - x)); // Reverse x for proper orientation
        if (z == 2 && mouthPattern[y][x]) // Adjust the Z-axis for mouth
          row |= (1 << (CUBE_SIZE_X - 1 - x)); // Reverse x for proper orientation
      }
      SPI.transfer(row); // Send the row byte
    }
    digitalWrite(SS, HIGH);
    delay(1); // Adjust this delay to control animation speed
  }
}

void setup() {
  // Initialize SPI communication
  SPI.beginTransaction(SPISettings(8000000, MSBFIRST, SPI_MODE0));
  // Initialize Slave Select pin
  pinMode(SS, OUTPUT);
}

void loop() {
  // Render the cube
  renderCube();
}
