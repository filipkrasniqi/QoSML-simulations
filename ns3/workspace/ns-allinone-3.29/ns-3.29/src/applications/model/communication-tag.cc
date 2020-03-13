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


#include <stdint.h>
#include "ns3/uinteger.h"
#include "communication-tag.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("CommunicationTag");

TypeId
 CommunicationTag::GetTypeId (void)
 {
   static TypeId tid = TypeId ("ns3::CommunicationTag")
     .SetParent<Tag> ()
     .AddConstructor<CommunicationTag> ()
     .AddAttribute ("CommValue",
                    "Communication value",
                    EmptyAttributeValue (),
                    MakeUintegerAccessor (&CommunicationTag::GetCommunication),
                    MakeUintegerChecker<uint64_t> ())
   ;
   return tid;
 }

 TypeId
 CommunicationTag::GetInstanceTypeId (void) const
 {
   return GetTypeId ();
 }

 uint32_t
 CommunicationTag::GetSerializedSize (void) const
 {
   return sizeof(uint64_t);
 }

 void
 CommunicationTag::Serialize (TagBuffer i) const
 {
   i.WriteU64 (m_communication);
 }

 void
 CommunicationTag::Deserialize (TagBuffer i)
 {
   m_communication = i.ReadU64 ();
 }

 void
 CommunicationTag::Print (std::ostream &os) const
 {
   os << "m_communication=" << (uint64_t)m_communication;
 }

 void
 CommunicationTag::SetCommunication (uint64_t communication)
 {
   m_communication = communication;
 }

 uint64_t
 CommunicationTag::GetCommunication (void) const
 {
   return m_communication;
 }
} // namespace ns3

