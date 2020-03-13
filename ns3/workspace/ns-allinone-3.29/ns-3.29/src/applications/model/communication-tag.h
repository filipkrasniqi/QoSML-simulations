/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Hajime Tazaki
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Hajime Tazaki <tazaki@sfc.wide.ad.jp>
 */

#ifndef COMMUNICATIONTAG_H
#define COMMUNICATIONTAG_H

#include "ns3/tag.h"
#include "ns3/uinteger.h"

namespace ns3 {


class Node;
class Packet;

/**
 * \ingroup ipv4
 *
 * \brief This class implements Linux struct pktinfo 
 * in order to deliver ancillary information to the socket interface.
 * This is used with socket option such as IP_PKTINFO, IP_RECVTTL, 
 * IP_RECVTOS. See linux manpage ip(7).
 *
 * This tag in the send direction is presently not enabled but we
 * would accept a patch along those lines in the future.
 */
class CommunicationTag : public Tag
{
 public:
   static TypeId GetTypeId (void);
   virtual TypeId GetInstanceTypeId (void) const;
   virtual uint32_t GetSerializedSize (void) const;
   virtual void Serialize (TagBuffer i) const;
   virtual void Deserialize (TagBuffer i);
   virtual void Print (std::ostream &os) const;

   // these are our accessors to our tag structure
   void SetCommunication (uint64_t communication);
   uint64_t GetCommunication (void) const;
 private:
   uint64_t m_communication;
};
} // namespace ns3

#endif /* IPV4_PACKET_INFO_TAG_H */
