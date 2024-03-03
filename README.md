# driver-drowsiness
int LED = 12;
int LPG_sensor = 3;
int LPG_detected;


void setup() {
  Serial.begin(9600);
  pinMode(LED, OUTPUT);
  pinMode(LPG_sensor, INPUT);
}

void loop() {
  LPG_detected = digitalRead(LPG_sensor);
  Serial.println(LPG_detected);

  if (LPG_detected == 1)
  {
    Serial.println("LPG detected...! take action immediately.");
    digitalWrite(LED, HIGH);
  }

  else
  {
    Serial.println("No detected, stay cool");
    digitalWrite(LED, LOW);
  }
}
