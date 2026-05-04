// Wifi setup
#include <SPI.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#define SendDeltaTimeInMs 100       // faster sends for mapping
#define ReceiveDeltaTimeInMs 10
#define NoSignalDeltaTimeInMs 2000

char ssid[] = "<Enter WiFi SSID>";
char pass[] = "<Enter WiFi Password>";
char remoteIP[] = "<Enter Laptop IP>";   // laptop IP
unsigned int localPort = 4010;           // Arduino listens here
unsigned int remotePort = 4010;          // laptop bridge listens here

int status = WL_IDLE_STATUS;
unsigned long last_time_rx = 0;
unsigned long last_time_tx = 0;

WiFiUDP Udp;
char packetBuffer[256];              // incoming control packets only

// Lidar setup
#include "RPLidar.h"
#define RPLidarMotorPin 3
#define NumLidarBins 180              // Total size with 180 bins:
#define LidarBinSizeDeg (360 / NumLidarBins)

RPLidar lidar;

int scan_ranges_mm[NumLidarBins];
bool scan_valid[NumLidarBins];
bool scan_ready = false;
int last_angle_deg = -1;


// -----------------------------------------------------------------------------
// Corridor-following controller configuration (line-fit + adaptive bias)
// -----------------------------------------------------------------------------
// User-defined "straight" steering command from teleop GUI.
const int TELEOP_STRAIGHT_STEER_CMD = -7;

// Match the mapper's LiDAR interpretation so "front", "left", and "right"
// mean the same thing in both the controller and the map.
const bool AUTO_SCAN_FLIP = true;
const float AUTO_ANGLE_ZERO_OFFSET_DEG = 0.0f;

// Steering command clamp.
const int STEER_CMD_MIN = -20;
const int STEER_CMD_MAX = 20;

// If positive correction steers the wrong physical direction, flip this sign.
const float AUTO_STEER_CORR_SIGN = -1.0f;

// Simple side-distance centering controller.
const int AUTO_LEFT_DEG = 90;
const int AUTO_RIGHT_DEG = 270;
const int AUTO_SIDE_WINDOW_HALF_WIDTH_DEG = 10;
const float AUTO_SIDE_RANGE_MIN_M = 0.15f;
const float AUTO_SIDE_RANGE_MAX_M = 1.80f;
const float AUTO_SIDE_DANGER_M = 0.38f;         // trigger recovery earlier
const float AUTO_SIDE_CAUTION_M = 0.55f;        // slow earlier near walls
const int AUTO_SIDE_RECOVER_STEER_CMD = 7;      // stronger steer away from wall
const float AUTO_EMERGENCY_SPEED_SCALE = 0.20f; // crawl during emergency recovery
const int AUTO_EMERGENCY_STEER_STEP_LIMIT_CMD = 3;

// Main centering gain.
const float AUTO_KY_CMD_PER_M = 7.0f;
const float AUTO_KI_CMD_PER_M_S = 2.0f;
const float AUTO_KV = 0.25f;

// Front safety.
const int AUTO_FRONT_DEG = 0;
const int AUTO_FRONT_WINDOW_HALF_WIDTH_DEG = 8;

// Speed modulation / safety.
const float AUTO_STOP_FRONT_M = 0.55f;
const float AUTO_SLOW_FRONT_M = 1.00f;
const float AUTO_MIN_SPEED_SCALE = 0.35f;

const int AUTO_MAX_TOTAL_CORR_CMD = 7;
const int AUTO_STEER_STEP_LIMIT_CMD = 1;
const int AUTO_FALLBACK_STEP_LIMIT_CMD = 1;

// Filtering / deadbands / learned bias bounds.
const float AUTO_CENTER_LPF_ALPHA = 0.20f;
const float AUTO_CENTER_DEADBAND_M = 0.02f;
const float AUTO_CORRIDOR_WIDTH_MIN_M = 0.50f;
const float AUTO_CORRIDOR_WIDTH_MAX_M = 2.20f;
const float AUTO_BIAS_CMD_MAX = 3.0f;

// Motor Control setup
#define RightSpeedPin 9
#define RightMotorDirPin1 12
#define RightMotorDirPin2 11
#define LeftSpeedPin 6
#define LeftMotorDirPin1 7
#define LeftMotorDirPin2 8

// Servo control setup
#include <Servo.h>
#define ServoPin 10
Servo myServo;

// Encoder setup
#define EncoderOutputA 4
#define EncoderOutputB 5
#define steering_angle_center 75

int a_state;
int encoder_a_last_state;
int encoder_count = 0;

// Structure for storing control signals received from laptop
struct ControlSignal {
  int speed = 0;
  int steering_angle = 0;
  int auto_corridor = 0;
};
ControlSignal last_control_signal;

// Structure for storing sensor signals sent to laptop
struct SensorSignal {
  int encoder_count = 0;
  int steering_angle = 0;
};
SensorSignal last_sensor_signal;

int last_applied_speed = 0;
int last_applied_steering = TELEOP_STRAIGHT_STEER_CMD;
float last_center_error_m = 0.0f;
float last_heading_error_rad = 0.0f;
float last_front_distance_m = -1.0f;
float filtered_center_error_m = 0.0f;
float filtered_heading_error_rad = 0.0f;
float learned_bias_cmd = 0.0f;
float last_corridor_width_m = -1.0f;
bool last_corridor_fit_ok = false;

// -----------------------------------------------------------------------------
// Adaptive controller state
// -----------------------------------------------------------------------------
unsigned long auto_prev_control_ms = 0;

// Compact binary scan packet.
// Total size with 360 bins:
// 4 bytes encoder + 2 bytes steering + 2 bytes num_bins + 360*2 bytes distances = 728 bytes
typedef struct __attribute__((packed)) {
  int32_t encoder_count;
  int16_t steering_angle;
  uint16_t num_bins;
  int16_t distances_mm[NumLidarBins];
} ScanPacket;

// Reset the scan bins
void reset_scan_bins() {
  for (int i = 0; i < NumLidarBins; i++) {
    scan_ranges_mm[i] = -1;
    scan_valid[i] = false;
  }
  scan_ready = false;
}

int clamp_int(int value, int low, int high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}


float clamp_float(float value, float low, float high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

float wrap_angle_rad(float angle) {
  while (angle > PI) angle -= 2.0f * PI;
  while (angle < -PI) angle += 2.0f * PI;
  return angle;
}

float wrap_angle_deg_360(float angle_deg) {
  while (angle_deg >= 360.0f) angle_deg -= 360.0f;
  while (angle_deg < 0.0f) angle_deg += 360.0f;
  return angle_deg;
}

float get_bin_center_deg_robot_frame(int bin_index) {
  float angle_deg = bin_index * LidarBinSizeDeg + 0.5f * LidarBinSizeDeg;

  if (AUTO_SCAN_FLIP) {
    angle_deg = 360.0f - angle_deg;
  }

  angle_deg += AUTO_ANGLE_ZERO_OFFSET_DEG;
  return wrap_angle_deg_360(angle_deg);
}

void reset_auto_controller_state() {
  auto_prev_control_ms = 0;
  filtered_center_error_m = 0.0f;
  filtered_heading_error_rad = 0.0f;
  last_center_error_m = 0.0f;
  last_heading_error_rad = 0.0f;
  last_front_distance_m = -1.0f;
  last_corridor_fit_ok = false;
  learned_bias_cmd = 0.0f;
}

void debug_link_state() {
  static unsigned long last_debug_ms = 0;
  if (millis() - last_debug_ms < 500) return;
  last_debug_ms = millis();

  Serial.print("WiFi.status=");
  Serial.print(WiFi.status());
  Serial.print(" RSSI=");
  Serial.print(WiFi.RSSI());
  Serial.print(" last_rx_age_ms=");
  Serial.print(millis() - last_time_rx);
  Serial.print(" last_tx_age_ms=");
  Serial.println(millis() - last_time_tx);
}

int angle_delta_deg(int a, int b) {
  int d = a - b;
  while (d > 180) d -= 360;
  while (d < -180) d += 360;
  return d;
}

bool average_range_m_for_angle_window(int target_deg, int half_width_deg, float &range_m) {
  long sum_mm = 0;
  int count = 0;

  for (int i = 0; i < NumLidarBins; i++) {
    if (!scan_valid[i] || scan_ranges_mm[i] <= 0) {
      continue;
    }

    int bin_center_deg = (int)round(get_bin_center_deg_robot_frame(i));
    if (abs(angle_delta_deg(bin_center_deg, target_deg)) <= half_width_deg) {
      sum_mm += scan_ranges_mm[i];
      count++;
    }
  }

  if (count == 0) {
    return false;
  }

  range_m = (sum_mm / (float)count) / 1000.0f;
  return true;
}

bool average_side_range_clipped_m(int target_deg, int half_width_deg, float &range_m) {
  float raw_m;
  if (!average_range_m_for_angle_window(target_deg, half_width_deg, raw_m)) {
    return false;
  }

  if (raw_m < AUTO_SIDE_RANGE_MIN_M || raw_m > AUTO_SIDE_RANGE_MAX_M) {
    return false;
  }

  range_m = raw_m;
  return true;
}

bool compute_corridor_command(
  const ControlSignal &input_cmd,
  int &out_speed,
  int &out_steer)
{
  out_speed = input_cmd.speed;
  out_steer = input_cmd.steering_angle;

  if (!input_cmd.auto_corridor || input_cmd.speed <= 0) {
    reset_auto_controller_state();
    return false;
  }

  unsigned long now_ms = millis();
  float dt_s = 0.05f;
  if (auto_prev_control_ms != 0) {
    dt_s = (now_ms - auto_prev_control_ms) / 1000.0f;
    dt_s = clamp_float(dt_s, 0.01f, 0.20f);
  }
  auto_prev_control_ms = now_ms;

  // ----------------------------------------------------------
  // Front safety
  // ----------------------------------------------------------
  float front_m;
  bool have_front = average_range_m_for_angle_window(AUTO_FRONT_DEG, AUTO_FRONT_WINDOW_HALF_WIDTH_DEG, front_m);
  last_front_distance_m = have_front ? front_m : -1.0f;

  if (have_front && front_m <= AUTO_STOP_FRONT_M) {
    out_speed = 0;
    out_steer = TELEOP_STRAIGHT_STEER_CMD;
    return true;
  }

  // ----------------------------------------------------------
  // Simple corridor centering from left/right side distances
  // ----------------------------------------------------------
  float left_m, right_m;
  bool have_left = average_side_range_clipped_m(AUTO_LEFT_DEG, AUTO_SIDE_WINDOW_HALF_WIDTH_DEG, left_m);
  bool have_right = average_side_range_clipped_m(AUTO_RIGHT_DEG, AUTO_SIDE_WINDOW_HALF_WIDTH_DEG, right_m);

  // ----------------------------------------------------------
  // Hard side-wall safety override
  // If too close to one wall, immediately steer away from it.
  // This takes priority over normal centering.
  // ----------------------------------------------------------
  bool danger_left = have_left && (left_m < AUTO_SIDE_DANGER_M);
  bool danger_right = have_right && (right_m < AUTO_SIDE_DANGER_M);
  bool caution_left = have_left && (left_m < AUTO_SIDE_CAUTION_M);
  bool caution_right = have_right && (right_m < AUTO_SIDE_CAUTION_M);

  if (danger_left && (!danger_right || left_m < right_m)) {
    int target_steer = clamp_int(
      TELEOP_STRAIGHT_STEER_CMD + AUTO_SIDE_RECOVER_STEER_CMD,
      STEER_CMD_MIN,
      STEER_CMD_MAX
    );

    int delta_steer = clamp_int(
      target_steer - last_applied_steering,
      -AUTO_EMERGENCY_STEER_STEP_LIMIT_CMD,
      AUTO_EMERGENCY_STEER_STEP_LIMIT_CMD
    );

    out_steer = clamp_int(last_applied_steering + delta_steer, STEER_CMD_MIN, STEER_CMD_MAX);
    out_speed = clamp_int((int)round(input_cmd.speed * AUTO_EMERGENCY_SPEED_SCALE), 0, 100);

    last_corridor_fit_ok = false;
    last_center_error_m = 0.0f;
    learned_bias_cmd = 0.0f;
    return true;
  }

  if (danger_right && (!danger_left || right_m < left_m)) {
    int target_steer = clamp_int(
      TELEOP_STRAIGHT_STEER_CMD - AUTO_SIDE_RECOVER_STEER_CMD,
      STEER_CMD_MIN,
      STEER_CMD_MAX
    );

    int delta_steer = clamp_int(
      target_steer - last_applied_steering,
      -AUTO_EMERGENCY_STEER_STEP_LIMIT_CMD,
      AUTO_EMERGENCY_STEER_STEP_LIMIT_CMD
    );

    out_steer = clamp_int(last_applied_steering + delta_steer, STEER_CMD_MIN, STEER_CMD_MAX);
    out_speed = clamp_int((int)round(input_cmd.speed * AUTO_EMERGENCY_SPEED_SCALE), 0, 100);

    last_corridor_fit_ok = false;
    last_center_error_m = 0.0f;
    learned_bias_cmd = 0.0f;
    return true;
  }

  bool corridor_ok = false;
  float center_error_m = 0.0f;

  if (have_left && have_right) {
    float corridor_width_m = left_m + right_m;
    last_corridor_width_m = corridor_width_m;

    if (corridor_width_m >= AUTO_CORRIDOR_WIDTH_MIN_M &&
        corridor_width_m <= AUTO_CORRIDOR_WIDTH_MAX_M) {
      corridor_ok = true;

      // Positive if left side is wider than right side.
      // AUTO_STEER_CORR_SIGN handles physical steering direction.
      center_error_m = 0.5f * (left_m - right_m);

      filtered_center_error_m =
        (1.0f - AUTO_CENTER_LPF_ALPHA) * filtered_center_error_m +
        AUTO_CENTER_LPF_ALPHA * center_error_m;

      center_error_m = filtered_center_error_m;

      if (fabs(center_error_m) < AUTO_CENTER_DEADBAND_M) {
        center_error_m = 0.0f;
      }
    }
  }

  last_corridor_fit_ok = corridor_ok;
  last_center_error_m = center_error_m;
  last_heading_error_rad = 0.0f;

  // Integral trim: this is what cancels constant drift bias automatically.
  if (corridor_ok) {
    learned_bias_cmd += AUTO_KI_CMD_PER_M_S * center_error_m * dt_s;
    learned_bias_cmd = clamp_float(learned_bias_cmd, -AUTO_BIAS_CMD_MAX, AUTO_BIAS_CMD_MAX);
  } else {
    // Slowly decay stale trim if side estimate is weak.
    learned_bias_cmd *= 0.95f;
  }

  float extra_cmd_f = 0.0f;
  if (corridor_ok) {
    extra_cmd_f =
      AUTO_STEER_CORR_SIGN *
      (AUTO_KY_CMD_PER_M * center_error_m + learned_bias_cmd);
  }

  int extra_cmd = clamp_int((int)round(extra_cmd_f), -AUTO_MAX_TOTAL_CORR_CMD, AUTO_MAX_TOTAL_CORR_CMD);

  // In auto mode, use the nominal straight command as the base.
  int target_steer = clamp_int(TELEOP_STRAIGHT_STEER_CMD + extra_cmd, STEER_CMD_MIN, STEER_CMD_MAX);

  // If corridor estimate is weak, just hold straight instead of inventing a correction.
  if (!corridor_ok) {
    target_steer = TELEOP_STRAIGHT_STEER_CMD;
  }

  int step_limit = corridor_ok ? AUTO_STEER_STEP_LIMIT_CMD : AUTO_FALLBACK_STEP_LIMIT_CMD;
  int delta_steer = target_steer - last_applied_steering;
  delta_steer = clamp_int(delta_steer, -step_limit, step_limit);
  out_steer = clamp_int(last_applied_steering + delta_steer, STEER_CMD_MIN, STEER_CMD_MAX);

  float speed_scale = corridor_ok ? 1.0f : 0.85f;
  speed_scale = min(speed_scale, 1.0f - AUTO_KV * (fabs(extra_cmd) / (float)AUTO_MAX_TOTAL_CORR_CMD));
  speed_scale = constrain(speed_scale, AUTO_MIN_SPEED_SCALE, 1.0f);

  if (caution_left || caution_right) {
    speed_scale = min(speed_scale, 0.55f);
  }

  if (have_front && front_m < AUTO_SLOW_FRONT_M) {
    float front_scale = (front_m - AUTO_STOP_FRONT_M) / (AUTO_SLOW_FRONT_M - AUTO_STOP_FRONT_M);
    front_scale = constrain(front_scale, AUTO_MIN_SPEED_SCALE, 1.0f);
    speed_scale = min(speed_scale, front_scale);
  }

  out_speed = clamp_int((int)round(input_cmd.speed * speed_scale), 0, 100);
  return true;
}

// Main setup to run on Arduino power up or reset
void setup() 
{
  Serial.begin(115200);
  Serial.println("Running robot base code!");

  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    while (true);
  }

  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);
    status = WiFi.begin(ssid, pass);
    delay(10000);
  }
  

  Serial.println("Connected to WiFi");
  printWifiStatus();
  Serial.println("\nStarted UDP...");
  Udp.begin(localPort);

  // LiDAR setup
  pinMode(RPLidarMotorPin, OUTPUT);
  analogWrite(RPLidarMotorPin, 255);
  Serial2.begin(460800);
  lidar.begin(Serial2);
  delay(1000);

  rplidar_response_device_info_t info;
  if (IS_OK(lidar.getDeviceInfo(info, 100))) {
    Serial.println("LiDAR detected");
    if (IS_OK(lidar.startScan())) {
      Serial.println("LiDAR scan started");
    } else {
      Serial.println("LiDAR scan start failed");
    }
  } else {
    Serial.println("LiDAR device NOT detected");
  }
  reset_scan_bins();

  // Motor setup
  pinMode(RightMotorDirPin1, OUTPUT);
  pinMode(RightMotorDirPin2, OUTPUT);
  pinMode(LeftSpeedPin, OUTPUT);
  pinMode(LeftMotorDirPin1, OUTPUT);
  pinMode(LeftMotorDirPin2, OUTPUT);
  pinMode(RightSpeedPin, OUTPUT);
  stop();

  // Servo setup
  myServo.attach(ServoPin);
  // Force the servo to the nominal straight-trim command on boot
  // and let it settle before driving.
  myServo.write(constrain(steering_angle_center + TELEOP_STRAIGHT_STEER_CMD, 0, 180));
  delay(700);
  
  // Encoder setup
  pinMode(EncoderOutputA, INPUT);
  pinMode(EncoderOutputB, INPUT);
  encoder_a_last_state = digitalRead(EncoderOutputA);

  last_time_rx = millis();
  last_time_tx = millis();
}
 
// Main loop
void loop() 
{
  ControlSignal control_signal = receive_control_signals(last_control_signal);
  last_control_signal = control_signal;

  control_robot(control_signal);

  SensorSignal sensor_signal = get_sensor_signal();
  send_sensor_signal(sensor_signal);
}

// Stop all robot motors
void stop()
{
  digitalWrite(RightMotorDirPin1, LOW);
  digitalWrite(RightMotorDirPin2, LOW);
  digitalWrite(LeftMotorDirPin1, LOW);
  digitalWrite(LeftMotorDirPin2, LOW);
}

// Drive robot forward
void forward(int speed)
{
  digitalWrite(RightMotorDirPin1, HIGH);
  digitalWrite(RightMotorDirPin2, LOW);
  digitalWrite(LeftMotorDirPin1, HIGH);
  digitalWrite(LeftMotorDirPin2, LOW);
  analogWrite(LeftSpeedPin, speed*0.85);
  analogWrite(RightSpeedPin, speed);
}

// Receive control signal messages from laptop
ControlSignal receive_control_signals(ControlSignal last_control_signal) {
  ControlSignal control_signal = last_control_signal;

  unsigned long new_time_rx = millis();
  if (new_time_rx - last_time_rx > ReceiveDeltaTimeInMs) {
    int packetSize = Udp.parsePacket();
    if (packetSize) {
      int len = Udp.read(packetBuffer, 255);
      if (len > 0) {
        packetBuffer[len] = 0;
      }
      control_signal = unpack_control_signal(packetBuffer);
      last_time_rx = new_time_rx;

      Serial.print("Received cmd: ");
      Serial.print(control_signal.speed);
      Serial.print(", ");
      Serial.print(control_signal.steering_angle);
      Serial.print(", auto=");
      Serial.println(control_signal.auto_corridor);
    }
  }

  if (new_time_rx - last_time_rx > NoSignalDeltaTimeInMs) {
    static unsigned long last_failsafe_print_ms = 0;

    control_signal.speed = 0;
    control_signal.steering_angle = TELEOP_STRAIGHT_STEER_CMD;
    control_signal.auto_corridor = 0;

    if (millis() - last_failsafe_print_ms > 500) {
      Serial.print("FAILSAFE STOP: no UDP control for ");
      Serial.print(new_time_rx - last_time_rx);
      Serial.print(" ms, WiFi.status=");
      Serial.print(WiFi.status());
      Serial.print(" RSSI=");
      Serial.println(WiFi.RSSI());
      last_failsafe_print_ms = millis();
    }
  }

  return control_signal;
}

// Update encoder and lidar data
SensorSignal get_sensor_signal() {
  encoder_update();
  lidar_update();

  // Send the actual applied steering command back to the mapper,
  // not just the raw GUI command.
  last_sensor_signal.steering_angle = last_applied_steering;
  last_sensor_signal.encoder_count = encoder_count;

  return last_sensor_signal;
}

// Get new lidar measurements and accumulate into 2-degree bins.
// When one revolution completes, mark scan_ready = true.
void lidar_update() {
  if (IS_OK(lidar.waitPoint())) {
    float distance = lidar.getCurrentPoint().distance;
    int angle_deg = int(lidar.getCurrentPoint().angle) % 360;

    // Detect wraparound from end of revolution to start of next
    if (last_angle_deg >= 0 && last_angle_deg > 300 && angle_deg < 60) {
      scan_ready = true;
    }
    last_angle_deg = angle_deg;

    if (distance > 100) {
      int bin = angle_deg / LidarBinSizeDeg;
      if (bin < 0) bin = 0;
      if (bin >= NumLidarBins) bin = NumLidarBins - 1;

      scan_ranges_mm[bin] = int(distance);
      scan_valid[bin] = true;
    }
  } else {
    static unsigned long last_lidar_restart_ms = 0;

    analogWrite(RPLidarMotorPin, 255);

    if (millis() - last_lidar_restart_ms > 1000) {
      Serial.println("LiDAR waitPoint timeout, restarting scan non-blocking");
      lidar.startScan();
      analogWrite(RPLidarMotorPin, 255);
      last_lidar_restart_ms = millis();
    }
  }
}

// Encoder update
void encoder_update() { 
  a_state = digitalRead(EncoderOutputA);

  if (a_state != encoder_a_last_state) {
    if (digitalRead(EncoderOutputB) != a_state) { 
      encoder_count++;
    } else {
      encoder_count--;
    }
  }

  encoder_a_last_state = a_state;
}

// Send a complete binary scan packet to laptop.
// Missing bins use -1.
void send_sensor_signal(SensorSignal sensor_signal)
{
  unsigned long new_time_tx = millis();

  if ((new_time_tx - last_time_tx > SendDeltaTimeInMs) && scan_ready) {
    ScanPacket pkt;
    pkt.encoder_count = (int32_t)sensor_signal.encoder_count;
    pkt.steering_angle = (int16_t)sensor_signal.steering_angle;
    pkt.num_bins = (uint16_t)NumLidarBins;

    for (int i = 0; i < NumLidarBins; i++) {
      if (scan_valid[i]) {
        pkt.distances_mm[i] = (int16_t)scan_ranges_mm[i];
      } else {
        pkt.distances_mm[i] = (int16_t)-1;
      }
    }

    int bp = Udp.beginPacket(remoteIP, remotePort);
    size_t written = Udp.write((const uint8_t*)&pkt, sizeof(pkt));
    int ep = Udp.endPacket();

    Serial.print("UDP bp=");
    Serial.print(bp);
    Serial.print(" written=");
    Serial.print(written);
    Serial.print(" end=");
    Serial.print(ep);
    Serial.print(" wifi=");
    Serial.println(WiFi.status());

    reset_scan_bins();
    last_time_tx = new_time_tx;
  }
}

// Control robot
void control_robot(ControlSignal control_signal){
  int applied_speed = control_signal.speed;
  int applied_steer = control_signal.steering_angle;

  compute_corridor_command(control_signal, applied_speed, applied_steer);

  if (applied_speed > 0) {
    forward(2 * applied_speed);
  } else {
    stop();
  }

  int desired_angle = constrain(steering_angle_center + applied_steer, 0, 180);
  myServo.write(desired_angle);

  last_applied_speed = applied_speed;
  last_applied_steering = applied_steer;
  static unsigned long last_debug_ms = 0;
  if (millis() - last_debug_ms > 250) {
    Serial.print("AUTO fit=");
    Serial.print(last_corridor_fit_ok ? 1 : 0);
    Serial.print(" width=");
    Serial.print(last_corridor_width_m, 3);
    Serial.print(" ey=");
    Serial.print(last_center_error_m, 3);
    Serial.print(" bias=");
    Serial.print(learned_bias_cmd, 3);
    Serial.print(" speed=");
    Serial.print(last_applied_speed);
    Serial.print(" applied=");
    Serial.println(last_applied_steering);
    last_debug_ms = millis();
  }
}

// Unpack control signal
ControlSignal unpack_control_signal(char* packed_control_signal_as_char) {
  ControlSignal control_signal;
  char* token;

  token = strtok(packed_control_signal_as_char, ",");
  if (token != NULL) {
    control_signal.speed = atof(token);
  }

  token = strtok(NULL, ",");
  if (token != NULL) {
    control_signal.steering_angle = atof(token);
  }

  token = strtok(NULL, ",");
  if (token != NULL) {
    control_signal.auto_corridor = atoi(token);
  } else {
    control_signal.auto_corridor = 0;
  }

  return control_signal;
}

// Print wifi status
void printWifiStatus() {
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}