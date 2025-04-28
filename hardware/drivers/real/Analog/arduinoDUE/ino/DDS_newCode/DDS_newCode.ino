// TODO
// 1) Add amplitude ramps (both directions?)
// 2) Add phase jumps
// 3) Add profile support

// INCLUDES
#include <SPI.h>
#include <Ethernet.h>
#include <stdlib.h>

//Define bool types
#define BOOL uint_fast8_t
#define FALSE 0
#define TRUE 1

// SETTINGS

//Serial Monitor Settings
#define SERMON	//Serial Monitor ON
//#undef SERMON	//Serial Monitor OFF

// Ethernet Settings

//eqm-dds1 DDSrf
byte mac[] = {0x00, 0xAA, 0xBB, 0xCC, 0xDC, 0x03 };
// byte mac[] = {0xA8, 0x61, 0x0A, 0xAE, 0x03, 0x9D };
// byte mac[] = {0xDE,0xAD,0xBE,0xEF,0xFE,0xED};
IPAddress ip(130,79,148,73);

//eqm-dds1 DDS920
//byte mac[] = {0x00, 0xAA, 0xBB, 0xCC, 0xDC, 0x01 };
//IPAddress ip(130,79,148,70);

//rydberg-dds7 DDS767
// byte mac[] = {0x00, 0xAA, 0xBB, 0xCC, 0xDA, 0x07 };
// IPAddress ip(130, 79, 148, 72);

//eqm-dds3 DDS575
//byte mac[] = {0x00, 0xAA, 0xBB, 0xCC, 0xDC, 0x02 };
//IPAddress ip(130,79,148,69);


// Ethernet Variables
EthernetServer server(80);

// LEDs
#define GREEN_LED 42
#define RED_LED 43

// DDS Settings
#define PLL_ENABLE TRUE
#define QUEUE_SIZE 4000		//Size of the DDS Queue
#define MAX_BLOCK_SIZE 40	//Max Size (in doubles) of one DDS Ramp Block
#define MAX_BLOCK_NUMBER 100	//Max Number of DDS Ramp Blocks

// DDS Variables

struct DDSQueueElement
{
  uint_fast8_t waitTrig;
  uint_fast32_t dataReg;
  uint_fast32_t addressReg;
};//end DDSQueueElement

const double reference_clock_frequency = 10e6;
const double default_output_frequency = 10e6;
const double target_pll_frequency = 2.5e9;
double system_clock_frequency;
unsigned int queueIndex = 0;
unsigned int previousQueueIdx = 0;
DDSQueueElement ddsQueue[QUEUE_SIZE];
double dataInput[MAX_BLOCK_SIZE * MAX_BLOCK_NUMBER];

void setup()
{
  //Serial Monitor
#ifdef SERMON
  Serial.begin(115200);
#endif // SERMON

  //LED Pin Setup
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, LOW);

  //DDS Communication Pins
  //						// DDS PIN	-	ARDUINO REGISTER
  //CONTROL PINS
  pinMode(12, OUTPUT);	// MPI00	-	D8
  // MPI01 - not connected - allways set to HIGH
  pinMode(2, OUTPUT);		// MPI02	-	B25

  //ADDRESS PINTS
  pinMode(25, OUTPUT);	// MPI08	-	D0
  pinMode(26, OUTPUT);	// MPI09	-	D1
  pinMode(27, OUTPUT);	// MPI10	-	D2
  pinMode(28, OUTPUT);	// MPI11	-	D3
  pinMode(14, OUTPUT);	// MPI12	-	D4
  pinMode(15, OUTPUT);	// MPI13	-	D5
  pinMode(29, OUTPUT);	// MPI14	-	D6
  pinMode(11, OUTPUT);	// MPI15	-	D7

  //DATA PINTS
  pinMode(34, OUTPUT);	// MPI16	-	C2
  pinMode(35, OUTPUT);	// MPI17	-	C3
  pinMode(36, OUTPUT);	// MPI18	-	C4
  pinMode(37, OUTPUT);	// MPI19	-	C5
  pinMode(38, OUTPUT);	// MPI20	-	C6
  pinMode(39, OUTPUT);	// MPI21	-	C7
  pinMode(40, OUTPUT);	// MPI22	-	C8
  pinMode(41, OUTPUT);	// MPI23	-	C9
  pinMode(51, OUTPUT);	// MPI24	-	C12
  pinMode(50, OUTPUT);	// MPI25	-	C13
  pinMode(49, OUTPUT);	// MPI26	-	C14
  pinMode(48, OUTPUT);	// MPI27	-	C15
  pinMode(47, OUTPUT);	// MPI28	-	C16
  pinMode(46, OUTPUT);	// MPI29	-	C17
  pinMode(45, OUTPUT);	// MPI30	-	C18
  pinMode(44, OUTPUT);	// MPI31	-	C19

  //RESET BUFFER PIN
  pinMode(31, OUTPUT);	// RESET	-	A7

  //EXT. AND INT. TRIGGER OF DDS
  pinMode(16, INPUT);		// EXT-TRIG	-	A13
  pinMode(24, OUTPUT);	// INT-TRIG	-	A15
  pinMode(23, OUTPUT);	// ENABLE	-	A14

  //Initial Values
  digitalWrite(23, LOW);	//EXT. TRIG OFF
  digitalWrite(24, LOW);	//INT. TRIG LOW
  digitalWrite(31, LOW);	//RESET OFF
  PIOB->PIO_ODSR = 0;		//B-REGISTER LOW

  //Ethernet Shield Setup
  Ethernet.begin(mac, ip);
  server.begin();
  Serial.print("*****alive*****");
  //Initialize DDS

  if (!ddsInit()) {
#ifdef SERMON
    Serial.println("DDS Initialization failed!");
#endif // SERMON
  }//if (!(ddsInit())

}//end setup



void loop() {
  //************************************************
  //**** BLOCK 1 - READ DATA AT TCP SERVER *********
  //************************************************

  // Connect Client to Server that is Reading data
  EthernetClient client = server.available();
  String incommingStr = "";

  // Read Data from Client
  if (client) {
    // Check if the client is connected OR bytes are avaible for reading
    while (client.connected()) {
      // Check if bytes are available for reading
      if (client.available()) {
        incommingStr += (char) client.read();
      }//end if client available
    }// end while client.connected()
#ifdef SERMON
    Serial.println("*****Incomming Data*****");
    Serial.print(incommingStr);
    Serial.println("\n*****End of Transmission*****");
#endif // SERMON
  }//end if client

  //Check if new Sequence is written or if DDS should reuse the old one
  if (incommingStr == "continue\n")
  { /* Reset queue idx to old value  */
    queueIndex = previousQueueIdx;
#ifdef SERMON
    Serial.println("Continue command received ...");
    Serial.println("... DDS is repeating the last uploaded sequence.");
#endif // SERMON

  }
  else { //incommingStr is not the Continue command -> Read new Sequence

    //************************************************
    //**** BLOCK 2 - EXTRACT DATA FROM STRING ********
    //************************************************
    int columnIdx = 0; //used to scan incomming String
    int blockIdx = 0; //used to scan incomming String
    if (incommingStr.length() > 0) {
#ifdef SERMON
      Serial.println("Extract Data from String ...");
#endif // SERMON

      //Scan through incommingStr String and extract Table
      int line = incommingStr.indexOf('\n'); //idx after current line
      int column = incommingStr.indexOf('\t'); //idx after current number
      while (line > 0) {
        columnIdx = 0;
        while (column < line && column > 0) {
          String number;
          //Get Current Number
          number = incommingStr.substring(0, column);
          //Remove "Number'\t'" from String
          incommingStr.remove(0, column + 1);
          //Convert String to Double
          if (columnIdx < MAX_BLOCK_SIZE && blockIdx < MAX_BLOCK_NUMBER) {
            dataInput[blockIdx * MAX_BLOCK_SIZE + columnIdx] = number.toDouble();
          }
          else {
#ifdef SERMON
            Serial.println("ERROR in loop(): dataInput array to small!");
#endif // SERMON
          }//if (i < MAX_BLOCK_SIZE && j < MAX_BLOCK_NUMBER)
#ifdef SERMON
          Serial.print(dataInput[blockIdx * MAX_BLOCK_SIZE + columnIdx]); Serial.print(" ");
#endif // SERMON
          //Search next Number
          column = incommingStr.indexOf('\t');
          line = incommingStr.indexOf('\n');
          ++columnIdx;
        }//while (incommingStr.indexOf('\t') > -1))
#ifdef SERMON
        Serial.print("\n");
#endif // SERMON
        //Remove '\n' from String
        incommingStr.remove(0, line + 1);
        //Search Next Position
        column = incommingStr.indexOf('\t');
        line = incommingStr.indexOf('\n');
        ++blockIdx;
      }//while (incommingStr.indexOf('\n') > 0)

#ifdef SERMON
      Serial.print("Data extracted successfully, received ");
      Serial.print(blockIdx); Serial.print(" Blocks of "); Serial.print(columnIdx);
      Serial.println(" columns each.");
#endif // SERMON
    }//if (incommingStr.length() > 0)
    //************************************************
    //**** BLOCK 3 - BUILD DDS QUEUE *****************
    //************************************************
    if (columnIdx > 0 && blockIdx > 0) {
#ifdef SERMON
      Serial.println("Building DDS Queue from received Ramp Block ...");
#endif // SERMON
      DDSRampDataToQueue(dataInput, min(columnIdx, MAX_BLOCK_SIZE), min(blockIdx, MAX_BLOCK_NUMBER));
#ifdef SERMON
      Serial.print("DDS Queue Finished, it contains ");
      Serial.print(queueIndex);
      Serial.println(" Elements!");
#endif // SERMON
      printQueue();
    }//if (columnIdx > 0 && blockIdx > 0)

  } //end if/else incommingStr = "continue"

  //************************************************
  //**** BLOCK 4 - RUN DDS QUEUE *******************
  //************************************************
  if (queueIndex > 0) {
    //Turn On the External Trigger
    extTrigOn();
#ifdef SERMON
    Serial.println("Starting DDS!");
#endif // SERMON

    //run the DDS Queue
    
    runDDSQueue();
    ddsIntTrigger();
    //Turn Off the External Trigger
    extTrigOff();
    ddsIntTrigger();
#ifdef SERMON
    Serial.println("DDS Finished!");
#endif // SERMON
    previousQueueIdx = queueIndex; //saveQueueIdx
    queueIndex = 0; // emptyQueue
  }//end if (queueIndex > 0)

}//end loop


//****************************************************************************************************************************
//******************************************************DDS FUNCTIONS*********************************************************
//****************************************************************************************************************************

void runDDSQueue() {
  //Run Through Queue
  for (int j = 0; j < queueIndex; ++j) {
    //Disable Writing
    PIOB->PIO_ODSR = 1 << 25;
    //Set Address
    PIOD->PIO_ODSR = ddsQueue[j].addressReg;
    //Set Data
    PIOC->PIO_ODSR = ddsQueue[j].dataReg;
    //Wait for Trigger High
    while (!getTrigSignal() && !ddsQueue[j].waitTrig) {}
    //Enable Writing
    PIOB->PIO_ODSR = 0;

    //Wait until Trigger Low Again
    while (getTrigSignal() && !ddsQueue[j].waitTrig) {}

  }//end for j
}//end runDDSQueue

inline uint_fast8_t getTrigSignal() {
  //Read Triger Register (PIN16 - A.13)
  return (PIOA->PIO_PDSR >> 13) & 1;
}//end getTrigSignal

void DDSRampDataToQueue(double* rampBlocks, unsigned int blockSize, unsigned int blockNumber) {
  //Fill DDS Queue according to the settings send via TCP
  for (int j = 0; j < blockNumber; j++) {
    if (rampBlocks[j * MAX_BLOCK_SIZE] == 0) //Single Frequency
      queueSingleFreq(rampBlocks[j * MAX_BLOCK_SIZE + 1], rampBlocks[j * MAX_BLOCK_SIZE + 5], rampBlocks[j * MAX_BLOCK_SIZE + 4], rampBlocks[j * MAX_BLOCK_SIZE + 6], rampBlocks[j * MAX_BLOCK_SIZE + 3]);
    else if (rampBlocks[j * MAX_BLOCK_SIZE] == 1)				//Frequency Ramp
      queueRamp(rampBlocks[j * MAX_BLOCK_SIZE + 1], rampBlocks[j * MAX_BLOCK_SIZE + 2], rampBlocks[j * MAX_BLOCK_SIZE + 3], rampBlocks[j * MAX_BLOCK_SIZE + 4], rampBlocks[j * MAX_BLOCK_SIZE + 7], rampBlocks[j * MAX_BLOCK_SIZE + 6], rampBlocks[j * MAX_BLOCK_SIZE + 5]);
    else if (rampBlocks[j * MAX_BLOCK_SIZE] == 2)				//Ramp and Jump
      queueRampJump(rampBlocks[j * MAX_BLOCK_SIZE + 1], rampBlocks[j * MAX_BLOCK_SIZE + 2], rampBlocks[j * MAX_BLOCK_SIZE + 3], rampBlocks[j * MAX_BLOCK_SIZE + 4], rampBlocks[j * MAX_BLOCK_SIZE + 5], rampBlocks[j * MAX_BLOCK_SIZE + 6], rampBlocks[j * MAX_BLOCK_SIZE + 9], rampBlocks[j * MAX_BLOCK_SIZE + 8], rampBlocks[j * MAX_BLOCK_SIZE + 7]);
    else if (rampBlocks[j * MAX_BLOCK_SIZE] == 3)				//Clear
    {
      if (j > 0 && rampBlocks[(j - 1)*MAX_BLOCK_SIZE] == 0)												//Before Clear Was Single Freq
        queueSingleFreq(rampBlocks[(j - 1)*MAX_BLOCK_SIZE + 1], 0, 1, 0, rampBlocks[j * MAX_BLOCK_SIZE + 1]);
      if (j > 0 && (rampBlocks[(j - 1)*MAX_BLOCK_SIZE] == 1 || rampBlocks[(j - 1)*MAX_BLOCK_SIZE] == 2))	//Before Clear Was Ramp
        queueSingleFreq(rampBlocks[(j - 1)*MAX_BLOCK_SIZE + 2], 0, 1, 0, rampBlocks[j * MAX_BLOCK_SIZE + 1]);
      if (j == 0)
        queueSingleFreq(default_output_frequency, 0, 1, 0, rampBlocks[j * MAX_BLOCK_SIZE + 1]);																			//Clear comes first
    }//end if Clear
    else if (rampBlocks[j * MAX_BLOCK_SIZE] == 4)				//Amplitude Ramp
      queueAmplitudeRamp(rampBlocks[j * MAX_BLOCK_SIZE + 1], rampBlocks[j * MAX_BLOCK_SIZE + 2] , rampBlocks[j * MAX_BLOCK_SIZE + 3], rampBlocks[j * MAX_BLOCK_SIZE + 4], rampBlocks[j * MAX_BLOCK_SIZE + 6], rampBlocks[j * MAX_BLOCK_SIZE + 7], rampBlocks[j * MAX_BLOCK_SIZE + 5]);
    else													//Unkown Block
    {
#ifdef SERMON
      Serial.println("ERROR in DDSRampDataToQueue: Unkown block identifier received!");
#endif // SERMON
    }//end else
  }//end for j
  //Disable Writing
  //Arduino waits on this element for the last trigger.
  ddsAddQueueElem(0x200, 0, 0);
}//end DDSRampBlockToQueue

void queueSingleFreq(const double &frequency, const double &amplitude, bool amplitudeChanged, const double &phase, uint_fast8_t noWait) {
#ifdef SERMON
  Serial.println("Adding Single Frequency to Queue!");
#endif // SERMON
  //Compute Frequency and Amplitude Bytes for DDS
  const uint_fast16_t DDSAmplitude(toDDSAmpl(amplitude));
  const uint_fast32_t DDSFrequency(toDDSFreq(frequency));
  const uint_fast16_t DDSPhase(toDDSPhase(phase));
  //Enable Profile Mode and Disable Ramp
  ddsToQueue8Bit(0x06, 1 << 7, noWait);
  //Write Amplitude Flag
  ddsToQueue8Bit(0x01, amplitudeChanged, 1);
  //Write Frequency
  ddsToQueue16Bit(0x2D, DDSFrequency, 1);
  ddsToQueue16Bit(0x2F, DDSFrequency >> 16, 1);
  //Write Amplitude if used
  if (amplitudeChanged) {
    ddsToQueue16Bit(0x33, DDSAmplitude, 1);
  }//if (amplitude Changed
  //Write Phase
  ddsToQueue16Bit(0x31, DDSPhase, 1);
}//end queueSingleFreq

void queueRamp(const double &startFreq, const double &endFreq,
               const double &freqStepSize, const double &freqStepRate, const double &amplitude, boolean amplitudeChanged, uint_fast8_t noWait) {
#ifdef SERMON
  Serial.println("Adding Frequency Ramp to Queue!");
  
#endif // SERMON

  //Compute Frequency, StepSize and Amplitude Words for DDS
  uint_fast32_t DDSFrequencyStart(toDDSFreq(startFreq));
  uint_fast32_t DDSFrequencyEnd(toDDSFreq(endFreq));
  uint_fast32_t DDSFrequencySteps(toDDSFreq(freqStepSize));
  const uint_fast16_t DDSAmplitude(toDDSAmpl(amplitude));

  //Convert Step Rate from seconds to DDS cycles
  const uint_fast16_t DDSStepRate = freqStepRate * system_clock_frequency / 24.0 + 0.3;

  //Enable Ramp Mode and disable Jumps
  ddsToQueue16Bit(0x06, 0x8809, noWait);

  //Set Autoclear and Enable Osk if amplitudeChanged == 1
  ddsToQueue8Bit(0x01, 0x40 + amplitudeChanged, 1);

  //Set Start Frequency
  ddsToQueue16Bit(0x11, DDSFrequencyStart, 1);
  ddsToQueue16Bit(0x13, DDSFrequencyStart >> 16, 1);

  //Set End Frequency
  ddsToQueue16Bit(0x15, DDSFrequencyEnd, 1);
  ddsToQueue16Bit(0x17, DDSFrequencyEnd >> 16, 1);

  //Set Frequency Step Size
  ddsToQueue16Bit(0x19, DDSFrequencySteps, 1);
  ddsToQueue16Bit(0x1B, DDSFrequencySteps >> 16, 1);
  
  Serial.println("frequency rate!");
  Serial.print(DDSStepRate);
  //Set Step Rate
  ddsToQueue16Bit(0x21, DDSStepRate, 1);

  //Write Amplitude if used
  if (amplitudeChanged) {
    ddsToQueue16Bit(0x33, DDSAmplitude, 1);
  }//if (amplitude Changed)
}//end queueRamp

void queueRampJump(const double &startFreq, const double &endFreq,
                   const double &freqStepSize, const double &freqStepRate, const double &startJumpFreq,
                   const double &endJumpFreq, const double &amplitude, boolean amplitudeChanged, uint_fast8_t noWait) {
#ifdef SERMON
  Serial.println("Adding Frequency Ramp and Jump to Queue!");
#endif // SERMON

  //Compute Frequency, StepSize and Amplitude Words for DDS
  uint_fast32_t DDSFrequencyStart(toDDSFreq(startFreq));
  uint_fast32_t DDSFrequencyEnd(toDDSFreq(endFreq));
  uint_fast32_t DDSFrequencySteps(toDDSFreq(freqStepSize));
  uint_fast32_t DDSJumpStart(toDDSFreq(startJumpFreq));
  uint_fast32_t DDSJumpEnd(toDDSFreq(endJumpFreq));
  const uint_fast16_t DDSAmplitude(toDDSAmpl(amplitude));

  //Convert Step Rate from seconds to DDS cycles
  const uint_fast16_t DDSStepRate = freqStepRate * system_clock_frequency / 24.0 + 0.3;

  //Enable Ramp Mode and Enable Jumps
  ddsToQueue16Bit(0x06, 0x8849, noWait);

  //Set Autoclear and Enable Osk if amplitudeChanged == 1
  ddsToQueue8Bit(0x01, 0x40 + amplitudeChanged, 1);

  //Set Start Frequency
  ddsToQueue16Bit(0x11, DDSFrequencyStart, 1);
  ddsToQueue16Bit(0x13, DDSFrequencyStart >> 16, 1);

  //Set End Frequency
  ddsToQueue16Bit(0x15, DDSFrequencyEnd, 1);
  ddsToQueue16Bit(0x17, DDSFrequencyEnd >> 16, 1);

  //Set Frequency Step Size
  ddsToQueue16Bit(0x19, DDSFrequencySteps, 1);
  ddsToQueue16Bit(0x1B, DDSFrequencySteps >> 16, 1);

  //Set Step Rate
  ddsToQueue16Bit(0x21, DDSStepRate, 1);

  //Set Jumps Start
  ddsToQueue16Bit(0x25, DDSJumpStart, 1);
  ddsToQueue16Bit(0x27, DDSJumpStart >> 16, 1);

  //Set Jump End
  ddsToQueue16Bit(0x29, DDSJumpEnd, 1);
  ddsToQueue16Bit(0x2B, DDSJumpEnd >> 16, 1);

  //Write Amplitude if used
  if (amplitudeChanged) {
    ddsToQueue16Bit(0x33, DDSAmplitude, 1);
  }//if (amplitude Changed)

}//end queueRampJump

void queueAmplitudeRamp(const double &startAmp, const double &endAmp,
                        const double &ampSteps, const double &ampStepRate, const double &frequency,
                        const double &phase, uint_fast8_t noWait)
{
#ifdef SERMON
  Serial.println("Adding Amplitude Ramp to Queue!");
#endif // SERMON
  //Compute Amplitude,Frequency and Phase Words for DDS
  const uint_fast32_t DDSFreq(toDDSFreq(frequency));
  const uint_fast16_t DDSPhase(toDDSPhase(phase));
  const uint_fast16_t DDSstartAmp(toDDSAmpl(startAmp));
  const uint_fast16_t DDSendAmp(toDDSAmpl(endAmp));

  //Compute Step Size
  const uint_fast16_t DDSAmpStepSize = toDDSAmpl(abs(((endAmp - startAmp) / ampSteps) + 0.5));

  //Compute Step Rate
  const uint_fast16_t DDSStepRate = ampStepRate * (system_clock_frequency / 24.0) + 0.3;

  //Enable Amplitude Ramp Mode, Disable Jumps
  ddsToQueue16Bit(0x06, 0xA808, noWait);

  //Set Autoclear and Enable Osk
  ddsToQueue16Bit(0x01, 0x41, 1);

  //Set Frequency
  ddsToQueue16Bit(0x2D, DDSFreq, 1);
  ddsToQueue16Bit(0x2F, DDSFreq >> 16, 1);

  //Set Phase
  ddsToQueue16Bit(0x31, DDSPhase, 1);

  //Start Amplitude
  ddsToQueue16Bit(0x11, DDSstartAmp, 1);
  ddsToQueue16Bit(0x13, 0x0000, 1);

  //End Amplitude
  ddsToQueue16Bit(0x15, DDSendAmp, 1);
  ddsToQueue16Bit(0x17, 0x0000, 1);

  //Set Amplitude Step Number
  ddsToQueue16Bit(0x19, DDSAmpStepSize, 1);
  ddsToQueue16Bit(0x1B, 0x0000, 1);

  //Set Step Rate
  ddsToQueue16Bit(0x21, DDSStepRate, 1);

}//end queueAmplitudeRamp

bool ddsInit() {

#ifdef SERMON
  Serial.println("Initialize DDS ...");
#endif // SERMON

  // Reset DDS via the Reset Pin
  digitalWrite(31, HIGH);
  delayMicroseconds(20);
  digitalWrite(31, LOW);

  system_clock_frequency = reference_clock_frequency;

  //Enable and configure PLL on the DDS if necessary
  if (PLL_ENABLE)
  {
    // Compute clock multiplier: f_pll = 2 * clock_multiplier * f_ref_clk
    const double clock_multiplier_double = target_pll_frequency / (2.0 * reference_clock_frequency);
    // The check has to be done with doubles because integeroverflows can easily happen with 8 bit uints
    if (clock_multiplier_double < 9.8 || clock_multiplier_double > 255.2) {
#ifdef SERMON
      Serial.println("In function ddsInit: PLL clock multiplier out of valid range 10 to 255");
#endif // SERMON
      return false;
    }//if (clock_multiplier_double < 9.8 || clock_multiplier_double > 255.2)
    const uint_fast8_t clock_multiplier_uint(clock_multiplier_double + .5);

    //Write PLL Settings to DDS
    ddsWrite8Bit(0x09, clock_multiplier_uint);	// Clock Multiplier
    ddsWrite8Bit(0x0A, 4);		// Enable PLL
    ddsWrite8Bit(0x03, 1);		// Start VCO calibration
    ddsIntTrigger();			// Send Trigger
    delayMicroseconds(50000);	// Wait for Calibration (>16ms)
    ddsWrite8Bit(0x03, 0);		// Stop VCO calibration
#ifdef SERMON
    Serial.println("VCO Callibrated!");
#endif // SERMON

    //Compute System Clock Frequency
    system_clock_frequency = 2.0 * double(clock_multiplier_uint) * reference_clock_frequency;
  } //if(PLL_ENABLE)

#ifdef SERMON
  Serial.print("System Clock Frequency: ");
  Serial.println(system_clock_frequency);
#endif // SERMON

  //DAC callibration
  ddsWrite8Bit(0x0F, 1);	//Start Calibration
  ddsIntTrigger();
  delayMicroseconds(152);	//Wait for Calibration
  ddsWrite8Bit(0x0F, 0);	//Stop Calibration
  ddsIntTrigger();
#ifdef SERMON
  Serial.println("DAC Callibrated!");
#endif // SERMON

  //Enable Profile Mode
  ddsWrite8Bit(0x06, 1 << 7);

  //Set default frequency for profile 0
  if (default_output_frequency * 2 > system_clock_frequency + .001) {
#ifdef SERMON
    Serial.println("In function ddsInit: Default output Frequency too large!");
#endif // SERMON
    return FALSE;
  } //if (default_output_frequency * 2 > system_clock_frequency + .001)
  const uint_fast32_t defaultFreq(toDDSFreq(default_output_frequency));
  ddsWrite8Bit(0x2C, defaultFreq);
  ddsWrite8Bit(0x2D, defaultFreq >> 8);
  ddsWrite8Bit(0x2E, defaultFreq >> 16);
  ddsWrite8Bit(0x2F, defaultFreq >> 24);
#ifdef SERMON
  Serial.print("DDS Default Frequency set to:");
  Serial.println(default_output_frequency);
#endif // SERMON
  ddsIntTrigger();
#ifdef SERMON
  Serial.println("DDS Initialization finished!");
#endif // SERMON
  return true;
}//end ddsInit

void ddsIntTrigger() {
  //Set IO-Out Pin HIGH
  digitalWrite(24, HIGH);
  //Wait for 1 syncclk period ( = 24E+6 / f_sysclk )
  const unsigned int wait_microseconds((24.0e6 / system_clock_frequency) + 5);
  delayMicroseconds(wait_microseconds);
  //Set IO-Out Pin LOW
  digitalWrite(24, LOW);
}//end ddsIoUpdate

inline void ddsWrite8Bit(const uint_fast8_t &address, const uint_fast8_t &data) {

  //Set all bits above b7 to 0
  uint_fast32_t fullData(data & 0x000000FF);
  uint_fast32_t fullAddress(address & 0x000000FF);
  //Shift Data Bits 2 to right to allign with C Register Pins C.2 to C.9
  fullData <<= 2;

  //Write Data to Register
  PIOB->PIO_ODSR = 1 << 25;		//Disable Writing
  PIOD->PIO_ODSR = fullAddress;	//Set Address
  PIOC->PIO_ODSR = fullData;	//Set Data
  PIOB->PIO_ODSR = 0;				//Start Writing
  //Wait 5 Cycles (This is Assembler Code)
  __asm__("nop\n\t""nop\n\t""nop\n\t""nop\n\t""nop\n\t");
  PIOB->PIO_ODSR = 1 << 25;		//Stop Writing
}//end ddsWrite8Bit

void ddsToQueue16Bit(const uint_fast8_t &upperAddress,
                     const uint_fast16_t &data, const uint_fast8_t &noWait) {
  //Set unused bits to 0
  uint_fast32_t dataLow(data & 0x000000FF);	//Lower 8 bits
  uint_fast32_t dataHigh(data & 0x0000FF00);	//Higher 8 bits
  uint_fast32_t fullAddress(upperAddress & 0x000000FF);
  //Shift bits to correct Register Positions
  dataLow <<= 2;	//Shift to C.2 to C.9
  dataHigh <<= 4;	//Shift to C.12 to C.19

  //Combine Data
  dataLow |= dataHigh;

  //Set 16 bit write mode (= D.8)
  fullAddress |= 0x00000100;

  //Add to Queue
  ddsAddQueueElem(fullAddress, dataLow, noWait);

}//end ddsToQueue16Bit

void ddsToQueue8Bit(const uint_fast8_t &address,
                    const uint_fast16_t &data, const uint_fast8_t &noWait) {
  //Set unused bits to 0
  uint_fast32_t dataLow(data & 0x000000FF);	//Lower 8 bits
  uint_fast32_t fullAddress(address & 0x000000FF);
  //Shift bits to correct Register Positions
  dataLow <<= 2;	//Shift to C.2 to C.9

  //Add to Queue
  ddsAddQueueElem(fullAddress, dataLow, noWait);
}//end ddsToQueue8Bit


inline uint_fast32_t toDDSFreq(const double &frequency) {
  return (uint_fast32_t) (frequency * 4294967296.0 / system_clock_frequency + 0.5);
}//end toDDSFreq

inline uint_fast16_t toDDSAmpl(const double &amplitude)
{
  return uint_fast16_t(amplitude * 4095);
}//end toDDSAmpl

inline uint_fast16_t toDDSPhase(const double &phase)
{
  return (uint_fast16_t) (65535 * (phase / 6.283185307) + 0.5);
}//end toDDSPhase

void ddsAddQueueElem(const uint_fast32_t &address,
                     const uint_fast32_t &data, const uint_fast8_t &waitTrig) {
  if (queueIndex < QUEUE_SIZE) {
    ddsQueue[queueIndex].addressReg = address;
    ddsQueue[queueIndex].dataReg = data;
    ddsQueue[queueIndex].waitTrig = waitTrig;
    ++queueIndex;
  } else {
#ifdef SERMON
    Serial.println("Error in Function ddsAddQueueElem: ddsQueue is full!");
#endif // SERMON
  } //if else (queueIndex < QUEUE_SIZE)
}//end ddsAddQueueElem

void printQueue() {
#ifdef SERMON
  Serial.println("****** Complete DDS Queue ******");
  for (int j = 0; j < queueIndex; j++) {
    Serial.print("(");
    Serial.print(ddsQueue[j].addressReg, HEX);
    Serial.print(", ");
    Serial.print(ddsQueue[j].dataReg, HEX);
    Serial.print(", ");
    Serial.print(ddsQueue[j].waitTrig, HEX);
    Serial.print(")\n");
  }//end for j
  Serial.println("****** End of DDS Queue ******");
#endif // SERMON
}//end printQueue

void extTrigOff() {
  digitalWrite(23, LOW);
  digitalWrite(RED_LED, LOW);
#ifdef SERMON
  Serial.println("External Trigger OFF!");
#endif // SERMON
}//end extTrigOff

void extTrigOn() {
  digitalWrite(23, HIGH);
  digitalWrite(RED_LED, HIGH);
#ifdef SERMON
  Serial.println("External Trigger ON!");
#endif // SERMON
}//end extTrigOn

//****************************************************************************************************************************
//****************************************************************************************************************************
