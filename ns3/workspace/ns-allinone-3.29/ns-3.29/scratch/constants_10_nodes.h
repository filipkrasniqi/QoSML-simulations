#ifndef CONSTS
  #define CONSTS

  /**
  * When application should start, aka, interval on which data is sent
  */
  const int START_APP = 0;
  /**
  * To make it coherent with the python part, 
  * end = START_APP + (PERIOD_LENGTH_SECONDS*NUM_OF_PERIODS)
  */
  const int PERIOD_LENGTH_SECONDS = 60;
  const int NUM_OF_PERIODS = 60;
  /**
  * Rest is configuration
  */
  const int NUM_OF_NODES = 10;
  const int NUM_LINKS = 12;
  const int NUM_COMMUNICATIONS = 12;
  const uint32_t PACKET_SIZE = 6000;
  const uint32_t N_PACKETS = 12000;
#endif
