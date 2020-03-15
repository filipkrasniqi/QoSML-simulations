/*
End-to-end Delay Prediction Based on Traffic Matrix Sampling. 
Filip Krasniqi, Jocelyne Elias, Jeremie Leguay, Alessandro E. C. Redondi.
IEEE INFOCOM WKSHPS - NI: The 3rd International Workshop on Network Intelligence. Toronto, July 2020.
*/

#include <fstream>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/tag.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include <iostream>
#include "ns3/traffic-control-module.h"

#include "ns3/flow-monitor.h"
#include "ns3/flow-monitor-helper.h"
//I redefined the communication tags because I need a tag when a communication starts to identify a packet.
#include "ns3/communication-tag.h"

#include <chrono>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

using namespace ns3;
using namespace std;

NS_LOG_COMPONENT_DEFINE ("Simulation");

uint32_t NUM_OF_NODES = 0;
uint32_t NUM_LINKS = 0;
const uint32_t MAX_NUM_LINKS = 256;
const uint32_t MAX_NUM_COMMUNICATIONS = 4096;

uint32_t NUM_OF_PERIODS = 0;
double PERIOD_LENGTH_SECONDS = 0;
uint32_t NUM_COMMUNICATIONS = 0;
double START_APP = 0;
/**
* Structure defining a measurement when related to delay
* (SendOutgoing, UnicastForward, LocalDeliver).
* bytes: effective packet size related to that transmission.
*        It may change from the one we define due to layers (fragmentation, add header, ...)
* time: timestamp on which the event of the related trace source occurs
* interface: output interface of the node on which the communication occurs.
*            Saved for log purposes and not really used.
* protocol: can be of value TCP_PROTOCOL, UDP_PROTOCOL or 1 in case of ACK to UDP
* sourceDestination: pair defining the OD flow. first ---> second
**/
typedef struct {
  uint32_t bytes;
  int64_t time;
  uint32_t interface;
  uint32_t protocol;
  std::pair<uint32_t,uint32_t> sourceDestination;
} Measurement;
/**
* Enum to represent whether the Transport is UDP, TCP or ICMP
**/
enum CommunicationType{UDP, TCP, ICMP};
/**
* Structure that defines the properties of an OD flow. Every instance has in common
* that the related ns3 application is an OnOffApplication.
* sourceDestination: pair of nodes ID that defines a flow first -> second
* type: enum defining the transport protocol. UDP, TCP or ICMP
* bytes: defines the size of a packet
* n_packets: defines the number of packets that will be sent during this flow.
*            Unlimited if 0.
* dataRate: value defining the interarrival rate. Check ns3 documentation to see
            how to define a correct rate. It can be done with array of char,
            with an uint32_t, ...
            I do Datarate d = DataRate(communication.dataRate)
**/
typedef struct {
  std::pair<uint32_t,uint32_t> sourceDestination;
  CommunicationType type;
  uint32_t bytes;
  uint32_t n_packets;
  uint32_t dataRate;
} Communication;
/**
* Structure that defines the properties of a link.
* link: defines which nodes are connected. pair<uint32_t,uint32_t> so to assign a pair
*       of IDs
* propagationDelay: propagation delay of the link
* dataRate: capacity of the link
* dropTailQueue: number of packets before the drop tail queue is overflown
* REDQueueSize: number of packets of RED (Random Early Detection).
*               Defined at Traffic Control layer
**/
typedef struct {
  std::pair<uint32_t,uint32_t> link;
  string propagationDelay;
  string dataRate;
  uint64_t dropTailQueue;
  uint64_t REDQueueSize;
} Link;
/**
* This data structure contains the mapping between the packet IDs and the measurements.
* Each packet ID will contain all the measurements that are done on the inner nodes.
* At every trace source related to delay, I need the information related to the
* origin, source and the nodes associated to the current load. In those trace sources,
* I fill the data structure with their measurement and then in the end I write everything
* on file. Note that in case of load, traffic and drop measurement this structure
* is not used to get the actual data, but for obtaining the OD flow given a packet ID
**/
std::map<uint64_t, std::vector<std::pair<uint32_t, Measurement>>> traffic;
/**
* This data structure contains the mapping between the node IDs and the addresses.
* Every node n is related to A(n) addresses. So, A(n) is the number of interfaces of node n.
* addressesPerNode[i] will contain all the A(i) addresses of node i
* I output this in a file so that, together with the routing file exported by ns3,
* I am able to define the routing policy as a n^4 vector.
* Check how Ipv4GlobalRoutingHelper::PrintRoutingTableAllAt() outputs
**/
map<uint32_t, vector<std::string>> addressesPerNode;
/**
* Structure just for log purposes. Allows to count tcp,udp and icmp packets
**/
typedef struct {
    uint32_t tcp;
    uint32_t udp;
    uint32_t icmp;
} CounterProtocol;
/**
* For log purposes, each time I add info for TM
**/
map<uint32_t, CounterProtocol> counterTcpUdpPerPacketSize;
/**
* Simulation time is defined in number of periods, given a fixed period.
* Each period is associated to a row in dataset, if .py and .cc are coherent.
**/
double simulation_time;
/**
* interfaceContainers is an array of Ipv4InterfaceContainer.
* Each Ipv4InterfaceContainer instance contains the address of the interfaces
* related to the corresponding link.
* Can be used to find the node ID given the address.
**/
Ipv4InterfaceContainer interfaceContainers[MAX_NUM_LINKS];
/**
* allLinks is the variable containing the instance of the links of the network.
**/
Link allLinks[MAX_NUM_LINKS];
/**
* communications is the variable containing the instance of the OD flows of the network.
**/
Communication communications[MAX_NUM_COMMUNICATIONS];
/**
* allNodes is the variable containing the instance of the nodes of the network.
**/
NodeContainer allNodes;
/**
* nodeContainers is the variable containing the instance of the nodes of the network
* for each link. At some point, when I use the PointToPointHelper to instance a p2p connection,
* it is required to have a NodeContainer containing the nodes of the p2p.
* This NodeContainer can be easily found in nodeContainers variable, as
* nodeContainers[i] contains the nodes of the i-th p2p.
**/
NodeContainer nodeContainers[MAX_NUM_LINKS];
/**
* When I do p2p install for the nodeContainers[i], a NetDeviceContainer is returned.
* I store the corresponding instance at deviceContainers[i], thus the need to have
* an array of the same size of nodeContainers. It will be used when assigning addresses.
**/
NetDeviceContainer deviceContainers[MAX_NUM_LINKS];
/**
* When I define a p2p connection I have the related QueueDiscipline, defined at
* TrafficControlLayer. The instance of the QueueDiscipline of i-th p2p connection
* can be found at qdiscs[i]
**/
QueueDiscContainer qdiscs[MAX_NUM_LINKS];
/**
* Need to define an ID for the communication. Starts from 0 and then it is increased
* every time a packet is enqueued.
**/
uint64_t global_communication_counter = 0;

/**
* A stream for each file.
* stream_enqueue/stream_dequeue are for delay.
* stream_dropped is for dropped packets.
* stream_tx/stream_rx are for TM and load.
**/
Ptr<OutputStreamWrapper> stream_enqueue;
Ptr<OutputStreamWrapper> stream_dequeue;
Ptr<OutputStreamWrapper> stream_dropped;
Ptr<OutputStreamWrapper> stream_tx;
Ptr<OutputStreamWrapper> stream_rx;
Ptr<OutputStreamWrapper> stream_addresses;

/**
* There is an integer in the header defining the transport protocol. This are the related values.
* I just need them to be defined to remind that are those. In my case I store the information
* with the enum because is the structure that better fits a finite set of values.
**/
const uint8_t TCP_PROTOCOL = 6;
const uint8_t UDP_PROTOCOL = 17;
const uint8_t ICMP_PROTOCOL = 1;

bool log_everything = false;
bool log_unusual = false;

/**
* Returns the CommunicationType given the integer read from the header
**/
static CommunicationType GetType(string protocol) {
  return protocol == "TCP" ? TCP : protocol == "UDP" ? UDP : ICMP;
}

/**
* Returns the CommunicationType given the integer read from the header
**/
static CommunicationType GetType(uint32_t protocol) {
  if(log_unusual && protocol != TCP_PROTOCOL && protocol != UDP_PROTOCOL && protocol != ICMP_PROTOCOL) {
    NS_LOG_UNCOND("Unusual CommunicationType::GetTypeString" << protocol);
  }
  return protocol == TCP_PROTOCOL ? TCP : protocol == UDP_PROTOCOL ? UDP : ICMP;
}

/**
* To print correctly in the file, I also need what I want to write in the file
* given the value returned in the header (string TCP or UDP)
**/
static std::string GetTypeString(uint32_t protocol) {
  if(log_unusual && protocol != TCP_PROTOCOL && protocol != UDP_PROTOCOL && protocol != ICMP_PROTOCOL) {
    NS_LOG_UNCOND("Unusual string::GetTypeString" << protocol);
  }
  return protocol == TCP_PROTOCOL ? "TCP" : protocol == UDP_PROTOCOL ? "UDP" : "ICMP";
}

/**
* Return node ID in container given the IP address.
* The information is in interfaceContainers, that contains for each instance of
* the array a pair of addresses. interfaceContainers[i] contains the i-th link
* in the form of a pair of addresses (interfaceContainers[i].GetAddress(0)
* or interfaceContainers[i].GetAddress(1)).
* So, the i-th position means that the corresponding nodeID is in allLinks[i].
* To know whether it is in allLinks[i].first or allLinks[i].second,
* the given address must match respectively
* interfaceContainers[i].GetAddress(0) or interfaceContainers[i].GetAddress(1).
**/
static uint32_t
RetrieveIndexNodeFromNodeAddress(Ipv4Address nodeAddress) {
  uint32_t i = 0;
  uint32_t returnNode = NUM_OF_NODES;
  while(returnNode >= NUM_OF_NODES && i < NUM_LINKS) {
      Ipv4Address firstInterface = interfaceContainers[i].GetAddress(0);
      if(firstInterface.Get() == nodeAddress.Get()) {
        returnNode = allLinks[i].link.first;
      } else {
        Ipv4Address secondInterface = interfaceContainers[i].GetAddress(1);
        if(secondInterface.Get() == nodeAddress.Get()) {
          returnNode = allLinks[i].link.second;
        }
      }
      i++;
  }

  NS_ASSERT_MSG(returnNode < NUM_OF_NODES, "Error in RetrieveIndexNodeFromNodeAddress");

  return returnNode;
}

/**
* Callback associated to UnicastForward trace source at L3
* (https://www.nsnam.org/doxygen/classns3_1_1_ipv4_l3_protocol.html#acc97efd317fd7e0c1a65c6247fa6537a).
* The Callback source is triggered when a forwarding is executed on a node
* that is neither an source nor a destination. Useful for per link delay
**/
static void
UnicastForward(uint32_t currentNode, const Ipv4Header &header, Ptr< const Packet > p, uint32_t interface) {
    int64_t time_in_nanoseconds = Simulator::Now().GetNanoSeconds();
    CommunicationTag cTag;
    bool tagResult = p->PeekPacketTag(cTag);
    NS_ASSERT_MSG(tagResult, "Error in UnicastForward");

    uint32_t communication_id = cTag.GetCommunication();
    uint32_t sourceID = RetrieveIndexNodeFromNodeAddress(header.GetSource());
    uint32_t destinationID = RetrieveIndexNodeFromNodeAddress(header.GetDestination());
    traffic[communication_id].push_back({currentNode, { p->GetSize(), time_in_nanoseconds, interface, header.GetProtocol(), {sourceID, destinationID} }});
}

/**
* Callback associated to SendOutgoing trace source at L3
* (https://www.nsnam.org/doxygen/classns3_1_1_ipv4_l3_protocol.html#a909297aa7ca87db2b7c91daefa2ed40a).
* The Callback is triggered when a source node enqueues on the output queue.
* If the queue is full, the packet is dropped right afterward.
* This is the first callback that is called, so here I assign
* the communication tag that will identify the packet. Useful for per link and e2e delay
**/
static void
SendOutgoing(uint32_t txNode, const Ipv4Header &header, Ptr< const Packet > p, uint32_t interface) {
    int64_t time_in_nanoseconds = Simulator::Now().GetNanoSeconds();
    uint64_t communication_id = global_communication_counter++;

    CommunicationTag cTag;

    bool tagResult = p->PeekPacketTag(cTag);
    NS_ASSERT_MSG(!tagResult, "Error in SendOutgoing");

    cTag.SetCommunication(communication_id);
    p->AddPacketTag (cTag);

    uint32_t sourceID = RetrieveIndexNodeFromNodeAddress(header.GetSource());
    uint32_t destinationID = RetrieveIndexNodeFromNodeAddress(header.GetDestination());

    traffic[communication_id] = { {txNode, { p->GetSize(), time_in_nanoseconds, interface, header.GetProtocol(), {sourceID, destinationID} } } };
}

/**
* Callback associated to LocalDeliver trace source at L3
* (https://www.nsnam.org/doxygen/classns3_1_1_ipv4_l3_protocol.html#a70adc805da9b70e8fb6f2a998f4df446).
* The Callback is triggered when a destination node dequeues the packet,
* so when the packet has effectively reached the destination.
* Useful for per link and e2e delay
**/
static void
LocalDeliver(uint32_t rxNode, const Ipv4Header &header, Ptr< const Packet > p, uint32_t interface) {
    int64_t time_in_nanoseconds = Simulator::Now().GetNanoSeconds();

    CommunicationTag cTag;

    bool tagResult = p->PeekPacketTag(cTag);
    NS_ASSERT_MSG(tagResult, "Error in LocalDeliver");

    uint64_t communication_id = cTag.GetCommunication();
    uint32_t sourceID = RetrieveIndexNodeFromNodeAddress(header.GetSource());
    uint32_t destinationID = RetrieveIndexNodeFromNodeAddress(header.GetDestination());
    traffic[communication_id].push_back({rxNode, { p->GetSize(), time_in_nanoseconds, interface, header.GetProtocol(), {sourceID, destinationID} }});
}

//variable working to check whether drops are traced correctly and eventually log it
uint32_t my_counter_drops = 0;
uint32_t global_counter_drops = 0;

/**
* Callback associated to Drop trace source at TrafficControlLayer.
* Please refer to the following example: https://www.nsnam.org/doxygen/traffic-control_8cc_source.html
* and to the following explanation: https://www.nsnam.org/docs/models/html/queue-discs.html
* In details, notice that, being that a TrafficControlLayer is present, the drops
* don't occur at other layers, because congestion is handled at this layer regardless
* from the transport protocol.
* The callback is triggered when a drop occurs on an interface
* Useful for e2e delay
**/
static void
DroppedPacket(uint32_t txNode, uint32_t rxNode, Ptr<const QueueDiscItem> p) {
  int64_t time_in_nanoseconds = Simulator::Now().GetNanoSeconds();
  Ptr<Packet> packet = p->GetPacket();

  CommunicationTag cTag;

  bool tagResult = packet->PeekPacketTag(cTag);
  NS_ASSERT_MSG(tagResult, "Error in DroppedPacket");

  uint32_t communication_id = cTag.GetCommunication();
  std::vector<std::pair<uint32_t, Measurement>> measurements = traffic[communication_id]; //measurement done so far
  std::pair<uint32_t, Measurement> firstMeasurement = measurements[0];
  uint32_t sourceID = firstMeasurement.second.sourceDestination.first;
  uint32_t destinationID = firstMeasurement.second.sourceDestination.second;
  std::string protocol = GetTypeString(firstMeasurement.second.protocol);
  std::string uid = "" + std::to_string(communication_id) + "_" + std::to_string(txNode) + "_" + std::to_string(rxNode);

  *stream_dropped->GetStream () << uid << "," << communication_id << "," << txNode << "," << rxNode << "," << time_in_nanoseconds << "," << sourceID << "_" << destinationID << "," << protocol << std::endl;
  my_counter_drops++;
}


/**
* Callback associated to MaxRx trace source at L2
* (https://www.nsnam.org/docs/release/3.18/doxygen/classns3_1_1_point_to_point_net_device.html#a1b4ac13e16c7028bbb1593f2fca53280).
* Useful for link load and traffic matrix
**/
static void
RxEnd (uint32_t rxNode, uint32_t txNode, Ptr<const Packet> p)
{
    int64_t time_in_nanoseconds = Simulator::Now().GetNanoSeconds();
    uint32_t bytes = p->GetSize();

    CommunicationTag cTag;
    bool tagResult = p->PeekPacketTag(cTag);
    NS_ASSERT_MSG(tagResult, "Error in RxEnd");

    uint64_t communication_id = cTag.GetCommunication();
    std::vector<std::pair<uint32_t, Measurement>> measurements = traffic[communication_id]; //measurement done so far
    std::pair<uint32_t, Measurement> firstMeasurement = measurements[0];
    uint32_t sourceID = firstMeasurement.second.sourceDestination.first;
    uint32_t destinationID = firstMeasurement.second.sourceDestination.second;
    std::string protocol = GetTypeString(firstMeasurement.second.protocol);

    std::string uid = "" + std::to_string(communication_id) + "_" + std::to_string(txNode) + "_" + std::to_string(rxNode);
    *stream_rx->GetStream () << uid << "," << communication_id << "," << txNode << "," << rxNode << "," << time_in_nanoseconds << "," << bytes << "," << sourceID << "_" << destinationID << "," <<protocol << std::endl;

    uint32_t count_tcp = firstMeasurement.second.protocol == TCP_PROTOCOL ? 1 : 0;
    uint32_t count_udp = firstMeasurement.second.protocol == UDP_PROTOCOL ? 1 : 0;
    uint32_t count_icmp = firstMeasurement.second.protocol == ICMP_PROTOCOL ? 1 : 0;

    if(counterTcpUdpPerPacketSize.find(bytes) == counterTcpUdpPerPacketSize.end()) {
      counterTcpUdpPerPacketSize[bytes] = {count_tcp, count_udp, count_icmp};
    } else {
      counterTcpUdpPerPacketSize[bytes].tcp += count_tcp;
      counterTcpUdpPerPacketSize[bytes].udp += count_udp;
      counterTcpUdpPerPacketSize[bytes].icmp += count_icmp;
    }
}

/**
* Callback associated to MacTx trace source at L2
* (https://www.nsnam.org/docs/release/3.18/doxygen/classns3_1_1_point_to_point_net_device.html#a1f915201c9a40e6221a61477590ddfef).
* Useful for link load and traffic matrix
**/
static void
TxBegin (uint32_t txNode, uint32_t rxNode, Ptr<const Packet> p)
{
    int64_t time_in_nanoseconds = Simulator::Now().GetNanoSeconds();
    uint32_t bytes = p->GetSize();  //probably useless

    CommunicationTag cTag;
    bool tagResult = p->PeekPacketTag(cTag);
    NS_ASSERT_MSG(tagResult, "Error in TxBegin");

    uint64_t communication_id = cTag.GetCommunication();
    std::vector<std::pair<uint32_t, Measurement>> measurements = traffic[communication_id]; //measurement done so far
    std::pair<uint32_t, Measurement> firstMeasurement = measurements[0];
    uint32_t sourceID = firstMeasurement.second.sourceDestination.first;
    uint32_t destinationID = firstMeasurement.second.sourceDestination.second;
    std::string protocol = GetTypeString(firstMeasurement.second.protocol);

    std::string uid = "" + std::to_string(communication_id) + "_" + std::to_string(txNode) + "_" + std::to_string(rxNode);
    *stream_tx->GetStream () << uid << "," << communication_id << "," << txNode << "," << rxNode << "," << time_in_nanoseconds << "," << bytes << "," << sourceID << "_" << destinationID << "," <<protocol << std::endl;
}

/**
* Compare two measurements by considering time
**/
bool sortMeasurementByTime(const pair<uint32_t, Measurement> &a,
              const pair<uint32_t, Measurement> &b)
{
    return (a.second.time < b.second.time);
}

/**
* Considers string as an iterator composed of values made by the delimiter.
* Used to iterate over the string while splitting it.
**/
string next_string(string& line, string delimiter = " ") {
  uint16_t position = line.find(delimiter);
  string returnValue = line.substr(0, position);
  line.erase(0, position + delimiter.length());
  return returnValue;
}

/**
* For log purposes: counts the global amount of drops
**/
static void
CountDrops(Ptr<const QueueDiscItem> p) {
  global_counter_drops++;
}

/**
* Own implementation of TCP app.
* The application sends a packet at a defined rate.
*/
class CustomSocketApp : public Application
{
public:
  CustomSocketApp ();
  virtual ~CustomSocketApp ();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  /**
  * Setup of the socket application.
  */
  void Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate);

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  void ScheduleTx (void);
  void SendPacket (void);

  /**
  * m_socket: actual socket that works at layer 4 to send data
  * m_peer: destination address
  * m_packetSize: packet size
  * m_nPackets: number of packets to be sent. If 0, unlimited
  * m_dataRate: timer will send the amount of data related
  *   to m_packetSize every 1/m_dataRate period
  * m_sendEvent: takes trace of current event sending something,
  *   so that when we require to stop the application we cancel the last packet
  * m_running: true if application is still on
  * m_packetsSent: takes trace of how many packets are sent
  *   (log purposes or useful when m_nPackets > 0)
  */
  Ptr<Socket>     m_socket;
  Address         m_peer;
  uint32_t        m_packetSize;
  uint32_t        m_nPackets;
  DataRate        m_dataRate;
  EventId         m_sendEvent;
  bool            m_running;
  uint32_t        m_packetsSent;
};

CustomSocketApp::CustomSocketApp ()
  : m_socket (0),
    m_peer (),
    m_packetSize (0),
    m_nPackets (0),
    m_dataRate (0),
    m_sendEvent (),
    m_running (false),
    m_packetsSent (0)
{
}

CustomSocketApp::~CustomSocketApp ()
{
  m_socket = 0;
}

/**
* Static instance of type id
**/
TypeId CustomSocketApp::GetTypeId (void)
{
  static TypeId tid = TypeId ("Socket application")
    .SetParent<Application> ()
    .SetGroupName ("Own applications")
    .AddConstructor<CustomSocketApp> ()
    ;
  return tid;
}

void
CustomSocketApp::Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate)
{
  m_socket = socket;
  m_peer = address;
  m_packetSize = packetSize;
  m_nPackets = nPackets;
  m_dataRate = dataRate;
}

/**
* Starts application.
* A packet is sent when we start the application.
*/
void
CustomSocketApp::StartApplication (void)
{
  m_running = true;
  m_packetsSent = 0;
  if (InetSocketAddress::IsMatchingType (m_peer))
    {
      m_socket->Bind ();
    }
  else
    {
      m_socket->Bind6 ();
    }
  m_socket->Connect (m_peer);
  SendPacket ();
}

/**
* Takes care of closing socket and canceling still sending packets
**/
void
CustomSocketApp::StopApplication (void)
{
  m_running = false;

  if (m_sendEvent.IsRunning ())
    {
      Simulator::Cancel (m_sendEvent);
    }

  if (m_socket)
    {
      m_socket->Close ();
    }
}

/**
* Sends packet and calls ScheduleTx to schedule next sending event according to m_dataRate
**/
void
CustomSocketApp::SendPacket (void)
{
  Ptr<Packet> packet = Create<Packet> (m_packetSize);
  m_socket->Send (packet);
  m_packetsSent++;
  if (m_packetsSent < m_nPackets || m_nPackets == 0)
    {
      ScheduleTx ();
    }
}

/**
* Actual scheduling of next tx according to datarate
**/
void
CustomSocketApp::ScheduleTx (void)
{
  if (m_running)
    {
      Time tNext (Seconds (m_packetSize * 8 / static_cast<double> (m_dataRate.GetBitRate ())));
      m_sendEvent = Simulator::Schedule (tNext, &CustomSocketApp::SendPacket, this);
    }
}

int
main (int argc, char *argv[])
{
  /**
  * For log purposes
  **/
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  /**
  * Program starts with some initializations, as it is meant to be when executed from
  * the generate_simulations.py.
  **/

  /**
  * Initialization of home directory. Check in identifier variables, if not present,
  * get it from pwuid
  **/
  const char *homedir;
  if ((homedir = getenv("HOME")) == NULL) {
      homedir = getpwuid(getuid())->pw_dir;
  }

  string homedir_str(homedir);

  /**
  * Identifiers of current simulation
  **/
  string topology = "";
  string identifier = "";
  string intensity_level = "";
  string capacity_number = "";
  string propagation_delay_number = "";

  string base_directory = homedir_str + "/ns3/workspace/ns-allinone-3.29/ns-3.29/";
  string base_output_directory = base_directory+"datasets/ns3/";
  ostringstream sstream;
  /**
  * The program will save the files defined in the streams at output_directory.
  **/
  CommandLine cmd;
  /**
  * These are the values to be passed in input.
  * assets = directory containing the simulation file. I will generate on assets/<ns3_output> the .tr files
  * Directory where to find assets: base_output_directory/topology/identifier/intensity_level/environment_<capacity_number>_<propagation_delay_number>
  **/
  cmd.AddValue ("topology", "Name of the topology", topology);                    //eg: abilene
  cmd.AddValue ("identifier", "Identifier for specific topology", identifier); //eg: v1 -> to differentiate different execution of generate_simulations for same topology
  cmd.AddValue ("intensity_level", "Level of intensity, i.e., number of rates combination", intensity_level);        //eg: intensity_0
  cmd.AddValue ("capacity_number", "Number of capacity combination", capacity_number);  //eg: 1
  cmd.AddValue ("propagation_delay_number", "Number of propagation delay combination", propagation_delay_number);  //eg: 1

  cmd.Parse (argc,argv);

  if(topology.empty()) {
      NS_LOG_UNCOND("Wrong topology");
      return -1;
  }

  if(identifier.empty()) {
      NS_LOG_UNCOND("Wrong identifier");
      return -1;
  }

  if(intensity_level.empty()) {
      NS_LOG_UNCOND("Wrong intensity_level");
      return -1;
  }

  if(capacity_number.empty()) {
      NS_LOG_UNCOND("Wrong capacity_number");
      return -1;
  }

  if(propagation_delay_number.empty()) {
      NS_LOG_UNCOND("Wrong propagation_delay_number");
      return -1;
  }

  /**
  * Assets dirs, i.e., where to save all the values given the information
  * to identify the current simulation
  **/
  string assets = base_output_directory+topology+"/"+identifier+"/"+intensity_level+"/environment_"+capacity_number+"_"+propagation_delay_number+"/";
  string output_directory = assets+"ns3_output/";
  //create dir at base_output_directory/topology/identifier/intensity_level/environment_<capacity_number>_<propagation_delay_number>
  char parser_command[2000];
  sprintf(parser_command, "mkdir -p %s", output_directory.c_str());
  int result = std::system(parser_command);

  NS_LOG_UNCOND("Executed " << parser_command << " to create dirs with result "<< result);

  string line;
  ifstream simulation_stream (assets+"simulation.txt");
  ifstream links_stream (assets+"links.txt");
  ifstream traffic_stream (assets+"traffic.txt");
  if(!simulation_stream.is_open()) {
      NS_LOG_UNCOND("Simulation file does not exist");
      return -1;
  }
  if(!links_stream.is_open()) {
      NS_LOG_UNCOND("Links file does not exist");
      return -1;
  }

  if(!traffic_stream.is_open()) {
      NS_LOG_UNCOND("Traffic file does not exist");
      return -1;
  }

  /**
  * Program contains now information to instance ns3 stuff.
  **/

  /**
  * Read from simulation file. It contains only a line on the format
  * START_APP PERIOD_LENGTH_SECONDS NUM_OF_PERIODS NUM_OF_NODES
  **/
  getline (simulation_stream,line);
  if(!simulation_stream) {
     NS_LOG_UNCOND("No value for traffic");
     return -1;
  }
  simulation_stream.close();

  START_APP = stod(next_string(line));
  PERIOD_LENGTH_SECONDS = stod(next_string(line));
  NUM_OF_PERIODS = stoi(next_string(line));
  NUM_OF_NODES = stoi(next_string(line));

  simulation_time = PERIOD_LENGTH_SECONDS * NUM_OF_PERIODS;
  /**
  * Links file contains only NUM_LINKS+1 lines on the format
  * First line: NUM_LINKS
  * Lines from 1 to NUM_LINKS+1:
  * first_node second_node propagationDelay capacity dropTailQueue REDQueueSize
  * being first_node, second_node the connected nodes
  **/
  getline (links_stream,line);
  if(!links_stream) {
     NS_LOG_UNCOND("No value for links");
     return -1;
  }
  NUM_LINKS = stoi(next_string(line));
  for(uint32_t i=0;i<NUM_LINKS;i++) {
      getline (links_stream,line);
      if(!links_stream) {
         NS_LOG_UNCOND("No value for links at line "<<i);
         return -1;
      }
      uint32_t first_node = stoi(next_string(line));
      uint32_t second_node = stoi(next_string(line));
      string propagationDelay = next_string(line);
      string capacity = next_string(line);
      uint64_t dropTailQueue = stoi(next_string(line));
      uint64_t REDQueueSize = stoi(next_string(line));
      if(log_everything) {
        NS_LOG_UNCOND(first_node << " " << second_node<< " " << propagationDelay<< " " << capacity<< " " << dropTailQueue<< " " << REDQueueSize);
      }
      std::pair<uint32_t,uint32_t> pair(first_node, second_node);
      allLinks[i] = {pair, propagationDelay, capacity, dropTailQueue, REDQueueSize};
  }
  links_stream.close();

  /**
  * Traffic file contains only NUM_COMMUNICATIONS+1 lines on the format
  * First line: NUM_COMMUNICATIONS
  * Lines from 1 to NUM_COMMUNICATIONS+1:
  * source destination type packet_size num_packets
  * num_packets can be 0. If so, the traffic continues during all the simulation,
  * regardless the number of packets that have already been sent (continuous flow)
  **/
  getline (traffic_stream,line);
  if(!traffic_stream) {
      NS_LOG_UNCOND("No value for traffic");
      return -1;
  }
  NUM_COMMUNICATIONS = stoi(next_string(line));
  for(uint32_t i=0;i<NUM_COMMUNICATIONS;i++) {
      getline (traffic_stream,line);
      if(!traffic_stream) {
          NS_LOG_UNCOND("No value for traffic at line "<<i);
          return -1;
      }
      uint32_t source = stoi(next_string(line));
      uint32_t destination = stoi(next_string(line));
      CommunicationType type = GetType(next_string(line));
      uint32_t packet_size = stoi(next_string(line));
      uint32_t num_packets = stoi(next_string(line));
      uint32_t arrivalRate = stoi(next_string(line));
      if(log_everything) {
        NS_LOG_UNCOND(source << " " << destination << " " << type<< " " << packet_size<< " " << num_packets << " " << arrivalRate);
      }
      std::pair<uint32_t,uint32_t> pair(source, destination);
      communications[i] = {pair, type, packet_size, num_packets, arrivalRate};
  }

  traffic_stream.close();

  //initialization of the output files
  AsciiTraceHelper asciiTraceHelper;
  sstream.str(std::string());
  sstream << output_directory << "enqueue.tr";
  stream_enqueue = asciiTraceHelper.CreateFileStream (sstream.str());
  *stream_enqueue->GetStream() << std::fixed;
  sstream.str(std::string());
  sstream << output_directory << "dequeue.tr";
  stream_dequeue = asciiTraceHelper.CreateFileStream (sstream.str());
  *stream_dequeue->GetStream() << std::fixed;
  sstream.str(std::string());
  sstream << output_directory << "dropped.tr";
  stream_dropped = asciiTraceHelper.CreateFileStream (sstream.str());
  sstream.str(std::string());
  sstream << output_directory << "tx.tr";
  stream_tx = asciiTraceHelper.CreateFileStream (sstream.str());
  sstream.str(std::string());
  sstream << output_directory << "rx.tr";
  stream_rx = asciiTraceHelper.CreateFileStream (sstream.str());

  //write header of output files
  *stream_enqueue->GetStream () << "UID,PID,TX,RX,START,O_D,PROTOCOL" << std::endl;
  *stream_dequeue->GetStream () << "UID,PID,TX,RX,END,SIZE,O_D,PROTOCOL" << std::endl;
  *stream_dropped->GetStream () << "UID,PID,TX,RX,TIME,O_D,PROTOCOL" << std::endl;
  *stream_tx->GetStream () << "UID,PID,TX,RX,START,SIZE,O_D,PROTOCOL" << std::endl;
  *stream_rx->GetStream () << "UID,PID,TX,RX,END,SIZE,O_D,PROTOCOL" << std::endl;

  /**
  * create a NodeContainer for each pair of nodes, as the class contains all
  * nodes of a network, being it
  * - wifi
  * - CSMA
  * - or p2p.
  * In our case, we have NUM_LINKS p2p connections. To do that, I need to instance
  * all nodes at first and then, on a list of NUM_LINKS instances of NodeContainer,
  * instead of calling create (that means a node is created) I call Add
  */
  allNodes.Create(NUM_OF_NODES);
  // set segment size to 1460. By default it is 590.
  Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue (1460));
  /**
  * Initialization of a NodeContainer instance for each link. This identifies
  * a PointToPoint connection, that is a network. It is needed when I do p2p install
  * as the PointToPointHelper requires to do .Install(nodeContainers[i]), so that it is able
  * to install a connection between the two nodes specified by the NodeContainer
  **/
  for(uint32_t i=0;i<NUM_LINKS;i++) {
    nodeContainers[i].Add(allNodes.Get(allLinks[i].link.first));
    nodeContainers[i].Add(allNodes.Get(allLinks[i].link.second));
  }

  /**
  * for each link, I define a trivial network from a p2p connection. To do that,
  * I iterate all the links and set the p2p properties. The helper allows to install the nodes properly.
  * TrafficControlLayer is set per link, so I do that at this point
  **/
  PointToPointHelper pointToPoint;
  TrafficControlHelper tch;
  tch.SetRootQueueDisc ("ns3::RedQueueDisc");

  //installation of internet stack. Necessarily to be put before TrafficControlHelper
  InternetStackHelper stack;
  stack.Install (allNodes);

  for(uint32_t i=0;i<NUM_LINKS;i++) {
    pointToPoint.SetQueue ("ns3::DropTailQueue",
      "MaxSize", QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, allLinks[i].dropTailQueue))
    );
    pointToPoint.SetChannelAttribute ("Delay", StringValue (allLinks[i].propagationDelay));
    pointToPoint.SetDeviceAttribute ("DataRate", StringValue (allLinks[i].dataRate));
    deviceContainers[i] = pointToPoint.Install(nodeContainers[i]);
    //setting TC = Traffic Control properties. In details, the queue discipline.
    qdiscs[i] = tch.Install (deviceContainers[i]);
    qdiscs[i].Get(0)->SetMaxSize(QueueSize (QueueSizeUnit::PACKETS, allLinks[i].REDQueueSize));
    qdiscs[i].Get(1)->SetMaxSize(QueueSize (QueueSizeUnit::PACKETS, allLinks[i].REDQueueSize));
  }

  //define addresses in a static way.
  Ipv4AddressHelper address;
  for(uint32_t i=0;i<NUM_LINKS;i++) {
    char baseAddress[10];
    sprintf(baseAddress, "10.1.%d.0", i);
    address.SetBase(baseAddress, "255.255.255.0");
    interfaceContainers[i] = address.Assign(deviceContainers[i]);
  }

  //now that I have addresses I save them in the map so that afterwards I log it

  //init addressesPerNode as empty for each node
  for(uint32_t i=0; i<NUM_OF_NODES;i++) {
    addressesPerNode[i] = {};
  }

  //fill addressesPerNode
  for(uint32_t i=0; i<NUM_LINKS;i++) {
      uint32_t absoluteIndex1 = allLinks[i].link.first;
      uint32_t absoluteIndex2 = allLinks[i].link.second;

      sstream.str(std::string());
      sstream << "" << interfaceContainers[i].GetAddress(0);
      addressesPerNode[absoluteIndex1].push_back(sstream.str());

      sstream.str(std::string());
      sstream << "" << interfaceContainers[i].GetAddress(1);
      addressesPerNode[absoluteIndex2].push_back(sstream.str());
  }

  //print all interfaceContainers (log purposes)
  for(uint32_t i=0;i<NUM_LINKS;i++) {
      if(log_everything) {
          NS_LOG_UNCOND("Link: " << allLinks[i].link.first << " -- "<<allLinks[i].link.second<<" corresponds to "<< interfaceContainers[i].GetAddress(0) << " " << interfaceContainers[i].GetAddress(1));
      }
  }

  /**
  * define the OD flows. I don't keep track of all the applications,
  * so I just install them after I set the properties related to the OnOffApp.
  * For each application, I need to define the address. Being that each link
  * is associated to a pair of node, that are identified with the id,
  * for the communication I need the address, so first thing that I do
  * is to search for the correct address, given a node. Not relevant which address
  * I choose, as long as the address is related to that node.
  **/
  Ptr<Socket> sockets[NUM_COMMUNICATIONS];
  uint16_t sinkPort = 8080;
  ApplicationContainer clientApps;
  for(uint32_t i=0;i<NUM_COMMUNICATIONS;i++) {
    uint32_t indexSender = communications[i].sourceDestination.first;
    uint32_t indexReceiver = communications[i].sourceDestination.second;

    //I can find the address properly by finding index i s.t. it has the same value for one of the links in allLinks as indexReceiver
    uint32_t interfaceIndex = 0;
    uint32_t addressInNetwork = 2; //either 0 or 1 because I have a PointToPoint. So, if 0, it is the first in the interface, otherwise it is the second
    while(addressInNetwork > 1 && interfaceIndex < NUM_LINKS) {
      if(allLinks[interfaceIndex].link.first == indexReceiver) {
        addressInNetwork = 0;
      } else if(allLinks[interfaceIndex].link.second == indexReceiver) {
        addressInNetwork = 1;
      } else {
        interfaceIndex++;
      }
    }

    NS_ASSERT_MSG(addressInNetwork < 2, "Wrong network");

    Ipv4Address interface = interfaceContainers[interfaceIndex].GetAddress(addressInNetwork);
    Address addressReceiver (InetSocketAddress (interface, sinkPort));

    //create the receiver
    PacketSinkHelper packetSinkHelper (communications[i].type == TCP ? "ns3::TcpSocketFactory" : "ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
    ApplicationContainer sinkAppReceiver = packetSinkHelper.Install (allNodes.Get (indexReceiver));
    sinkAppReceiver.Start (Seconds (START_APP));

    if(communications[i].type == TCP) {
      /**
      * TCP socket application using our own implementation.
      **/
      Ptr<Socket> ns3TcpSocket = Socket::CreateSocket (allNodes.Get(indexSender), TcpSocketFactory::GetTypeId ());
      Ptr<CustomSocketApp> app = CreateObject<CustomSocketApp> ();
      app->Setup (ns3TcpSocket, addressReceiver, communications[i].bytes, communications[i].n_packets, DataRate (communications[i].dataRate));
      allNodes.Get(indexSender)->AddApplication (app);
      app->SetStartTime (Seconds (START_APP));
    } else {
      /**
      * UDP CBR using OnOffHelper. eg: https://www.nsnam.org/doxygen/simple-global-routing_8cc_source.html
      **/
      OnOffHelper clientHelper ("ns3::UdpSocketFactory", addressReceiver);
      clientHelper.SetAttribute("PacketSize", UintegerValue(communications[i].bytes));
      DataRate rate = DataRate(communications[i].dataRate);
      clientHelper.SetConstantRate (rate);
      clientApps = clientHelper.Install (allNodes.Get(indexSender));
    }

    clientApps.Start (Seconds (START_APP));
    sinkPort++;
  }

  /**
  * routing using Dijkstra for shortest path.
  * Every node is a router and allows to route to the neighbours
  **/
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  /**
  * Network configuration is finished. Now I define the trace sources.
  **/

  //for each node I define the queuing, dequeuing and forwarding listeners
  for(uint32_t i=0; i<NUM_OF_NODES;i++) {
      char connectionStringSO[60];
      sprintf(connectionStringSO, "/NodeList/%d/$ns3::Ipv4L3Protocol/SendOutgoing", i);
      char connectionStringLD[60];
      sprintf(connectionStringLD, "/NodeList/%d/$ns3::Ipv4L3Protocol/LocalDeliver", i);
      char connectionStringUF[60];
      sprintf(connectionStringUF, "/NodeList/%d/$ns3::Ipv4L3Protocol/UnicastForward", i);

      Config::ConnectWithoutContext(connectionStringSO, MakeBoundCallback(&SendOutgoing, i));
      Config::ConnectWithoutContext(connectionStringLD, MakeBoundCallback(&LocalDeliver, i));
      Config::ConnectWithoutContext(connectionStringUF, MakeBoundCallback(&UnicastForward, i));
  }

  // for each link I have 2 outgoing queues (once a traffic on one way is defined, eg: ACK will communicate also on the other side).
  // For each outgoing queue, I define the drop listener, that is set on the TC layer
  for(uint32_t i=0; i<NUM_LINKS;i++) {
    uint32_t absoluteIndex1 = allLinks[i].link.first;
    uint32_t absoluteIndex2 = allLinks[i].link.second;
    uint32_t interfaceIndex1 = deviceContainers[i].Get(0)->GetIfIndex();
    uint32_t interfaceIndex2 = deviceContainers[i].Get(1)->GetIfIndex();
    char connectionStringDrop[80];
    sprintf(connectionStringDrop, "/NodeList/%d/$ns3::TrafficControlLayer/RootQueueDiscList/%d/Drop", absoluteIndex1, interfaceIndex1);
    char connectionStringDrop2[80];
    sprintf(connectionStringDrop2, "/NodeList/%d/$ns3::TrafficControlLayer/RootQueueDiscList/%d/Drop", absoluteIndex2, interfaceIndex2);

    Config::ConnectWithoutContext (connectionStringDrop, MakeBoundCallback (&DroppedPacket, absoluteIndex1, absoluteIndex2));
    Config::ConnectWithoutContext (connectionStringDrop2, MakeBoundCallback (&DroppedPacket, absoluteIndex2, absoluteIndex1));

    deviceContainers[i].Get (0)->TraceConnectWithoutContext ("MacRx", MakeBoundCallback (&RxEnd,absoluteIndex1, absoluteIndex2));
    deviceContainers[i].Get (1)->TraceConnectWithoutContext ("MacRx", MakeBoundCallback (&RxEnd,absoluteIndex2, absoluteIndex1));
    deviceContainers[i].Get (0)->TraceConnectWithoutContext ("MacTx", MakeBoundCallback (&TxBegin, absoluteIndex1, absoluteIndex2));
    deviceContainers[i].Get (1)->TraceConnectWithoutContext ("MacTx", MakeBoundCallback (&TxBegin, absoluteIndex2, absoluteIndex1));
  }

  // only for log purposes to check whether there is a difference between drops defined as above and in a global way. It has never been the case.
  Config::ConnectWithoutContext ("/NodeList/*/$ns3::TrafficControlLayer/RootQueueDiscList/*/Drop", MakeCallback (&CountDrops));

  //writing the routing (log purposes)
  Ipv4GlobalRoutingHelper g;

  sstream.str(std::string());
  sstream << output_directory << "routing.routes";

  Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper> (sstream.str(), std::ios::out);

  uint32_t routing_print_time = 2;  // after 2 ns3 secs I check the routing
  g.PrintRoutingTableAllAt (Seconds (routing_print_time), routingStream);

  //writing the addresses per node
  sstream.str(std::string());
  sstream << output_directory << "addresses.tr";
  stream_addresses = asciiTraceHelper.CreateFileStream (sstream.str());
  for(uint32_t i=0;i<addressesPerNode.size();i++) {
    *stream_addresses->GetStream () << addressesPerNode[i].size() << " ";
    for(uint32_t j=0;j<addressesPerNode[i].size();j++) {
      *stream_addresses->GetStream () << addressesPerNode[i][j] << " ";
    }
    *stream_addresses->GetStream () << std::endl;
  }

  //init the flow monitor (log purposes)
  Ptr<FlowMonitor> flowMonitor;
  FlowMonitorHelper flowHelper;
  flowMonitor = flowHelper.InstallAll();

  const int DELTA = 1;
  //execute simulation
  Simulator::Stop (Seconds (START_APP + simulation_time + DELTA));
  Simulator::Run ();
  Simulator::Destroy ();
  /**
  * At this point the simulation has finished.
  * Trace files related to Drops, TM: already ok
  * Trace files related to delay: need to write to files using the mapping
  **/

  //writing flow monitor
  flowMonitor->SerializeToXmlFile("flowmonitor.xml", true, true);
  std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

  std::cout << "Time needed to execute the simulation  = " << std::chrono::duration<double>(end - begin).count() <<std::endl;

  if(log_everything) {
    for(uint64_t i=0;i<traffic.size();i++) {
      for(std::pair<uint32_t,Measurement> measurement : traffic[i]) {
        NS_LOG_UNCOND(i << " " << measurement.first << " " << measurement.second.time << " " << measurement.second.interface << " "
        << measurement.second.bytes << " " << measurement.second.protocol << " " << GetType(measurement.second.protocol) << " "
        << measurement.second.sourceDestination.first << " " << measurement.second.sourceDestination.second);
      }
    }
  }

  // once simulation is finished, I have already written for the Traffic Matrix and the drops.
  // I need to write the data for the delay, that are in the map data structure

  /**
  * Logging part related to tx/rx. I need the data structure because I didn't have
  * any way to know which was tx/rx. Instead of looking on the routing table,
  * I have the data structure.
  * I fill enqueue and dequeue streams
  */
  for(uint64_t i=0;i<traffic.size();i++) {
    //sort in case the vector is not ordered
    sort(traffic[i].begin(), traffic[i].end(), sortMeasurementByTime);
    for(uint32_t j = 0; j < traffic[i].size(); j++) {
      std::pair<uint32_t,Measurement> measurement = traffic[i][j];
      if(j < traffic[i].size()-1) {
          // this is the case on which I am enqueuing, i.e., all cases except the last one.
          std::pair<uint32_t,Measurement> nextMeasurement = traffic[i][j+1];
          uint32_t txNode = measurement.first;
          uint32_t rxNode = nextMeasurement.first;
          uint32_t sourceNode = measurement.second.sourceDestination.first;
          uint32_t destinationNode = measurement.second.sourceDestination.second;
          std::string protocol = GetTypeString(measurement.second.protocol);
          std::string uid = "" + std::to_string(i) + "_" + std::to_string(txNode) + "_" + std::to_string(rxNode);
          *stream_enqueue->GetStream () << uid << "," << i << "," << txNode << "," << rxNode << "," << measurement.second.time << "," << sourceNode << "_" << destinationNode << "," <<protocol << std::endl;
      }
      if(j > 0) {
        // this is the case on which I am dequeuing, i.e., all cases except the first one
        std::pair<uint32_t,Measurement> previousMeasurement = traffic[i][j-1];
        uint32_t txNode = previousMeasurement.first;
        uint32_t rxNode = measurement.first;
        uint32_t sourceNode = measurement.second.sourceDestination.first;
        uint32_t destinationNode = measurement.second.sourceDestination.second;
        std::string protocol = GetTypeString(measurement.second.protocol);
        std::string uid = "" + std::to_string(i) + "_" + std::to_string(txNode) + "_" + std::to_string(rxNode);

        *stream_dequeue->GetStream () << uid << "," << i << "," << txNode << "," << rxNode << "," << measurement.second.time << "," << measurement.second.bytes << "," << sourceNode << "_" << destinationNode << "," <<protocol << std::endl;
      }
    }
  }

  // next two blocks are only for log purposes
  if(log_everything) {
      NS_LOG_UNCOND("Found "<<counterTcpUdpPerPacketSize.size()<<" different sizes");

      for(std::map<uint32_t,CounterProtocol>::iterator iter = counterTcpUdpPerPacketSize.begin(); iter != counterTcpUdpPerPacketSize.end(); ++iter)
      {
        uint32_t size =  iter->first;
        CounterProtocol counters = iter->second;
        NS_LOG_UNCOND("Size "<<size<<" has "<< counters.tcp<<" TCP, "<<counters.udp<<" UDP AND "<<counters.icmp<<" ICMP");
      }
  }

  if(log_unusual && (my_counter_drops == 0 || global_counter_drops == 0 || my_counter_drops != global_counter_drops)) {
    NS_LOG_UNCOND("Unusual difference between count of drops: "<< global_counter_drops << " " <<my_counter_drops);
  }

  // build string to execute build_dataset
  sprintf(parser_command, "python %sbuild_dataset.py %d %d %d 1 %lf P %s %s %s", base_directory.c_str(), NUM_OF_PERIODS, NUM_OF_NODES, NUM_OF_NODES, PERIOD_LENGTH_SECONDS, "ns3_output", assets.c_str(), "datasets");
  NS_LOG_UNCOND("Simulation completed. Executing " << parser_command << " to build the dataset...");

  // actually execute the python code to build the dataset
  result = std::system(parser_command);

  NS_LOG_UNCOND("Result of execution: " << result);

  return 0;
}
