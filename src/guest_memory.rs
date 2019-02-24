// Copyright (C) 2019 Alibaba Cloud Computing. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

//! Traits to track and access memory regions that are mapped to a virtual machine.
//!
//! The GuestMemoryRegion trait is used to represent a continuous region of guest's physical memory.
//! And the GuestMemory trait is used to represent a collection of GuestMemoryRegion objects.
//! The main responsibilities of the GuestMemory trait are:
//! - hide the detail of accessing guest's physical address.
//! - map a request address to a GuestMemoryRegion object and relay the request to it.
//! - handle cases where an access request spanning two or more GuestMemoryRegion objects.

use address_space::{Address, AddressRegion, AddressSpace, AddressValue};
use std::fmt::{self, Display};
use std::io;
use std::ops::{BitAnd, BitOr};

/// Errors associated with handling guest memory accesses.
#[allow(missing_docs)]
#[derive(Debug)]
pub enum Error {
    /// No memory region found.
    NoMemoryRegion,
    /// Some of the memory regions intersect with each other.
    MemoryRegionOverlap,
    /// Failure in finding a guest address in any memory regions mapped by this guest.
    InvalidGuestAddress(GuestAddress),
    /// Failure in finding a guest address range in any memory regions mapped by this guest.
    InvalidGuestAddressRange(GuestAddress, GuestAddressValue),
    /// Writing to memory failed
    WriteToMemory(io::Error),
    /// Reading from memory failed
    ReadFromMemory(io::Error),
    /// Couldn't write to the given target.
    WriteToTarget(io::Error),
    /// Couldn't read from the given source.
    ReadFromSource(io::Error),
    /// Incomplete write
    ShortWrite {
        expected: GuestAddressValue,
        completed: GuestAddressValue,
    },
    /// Incomplete read
    ShortRead {
        expected: GuestAddressValue,
        completed: GuestAddressValue,
    },

    /// Syscall returned the given error.
    SystemCallFailed(io::Error),
    /// Requested backend address is out of range.
    InvalidBackendAddress,
    /// Requested offset is out of range.
    InvalidBackendOffset,
    /// Backend doesn't support the operation.
    InvalidBackendOperation,
}

impl std::error::Error for Error {}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Guest memory error: ")?;
        match self {
            Error::NoMemoryRegion => write!(f, "no region found"),
            Error::MemoryRegionOverlap => write!(f, "overlapping memory regions"),
            Error::InvalidGuestAddress(addr) => {
                write!(f, "invalid guest address {}", addr.raw_value())
            }
            Error::InvalidGuestAddressRange(base, size) => write!(
                f,
                "invalid address range, base {}/size {}",
                base.raw_value(),
                size,
            ),
            Error::WriteToMemory(_) => write!(f, "couldn't write to memory"),
            Error::ReadFromMemory(_) => write!(f, "couldn't read from memory"),
            Error::WriteToTarget(_) => write!(f, "couldn't write to target"),
            Error::ReadFromSource(_) => write!(f, "couldn't read from source"),
            Error::ShortWrite {
                expected,
                completed,
            } => write!(
                f,
                "incomplete write of {} instead of {} bytes",
                completed, expected,
            ),
            Error::ShortRead {
                expected,
                completed,
            } => write!(
                f,
                "incomplete read of {} instead of {} bytes",
                completed, expected,
            ),
            Error::SystemCallFailed(e) => write!(f, "syscall failed due to {}", e),
            Error::InvalidBackendAddress => write!(f, "invalid backend address"),
            Error::InvalidBackendOffset => write!(f, "invalid backend offset"),
            Error::InvalidBackendOperation => write!(f, "invalid backend operation"),
        }
    }
}

/// Represents a guest physical address (GPA).
///
/// Notes:
/// - On ARM64, a 32-bit hypervisor may be used to support a 64-bit guest. For simplicity,
/// u64 is used to store the the raw value no matter the guest a 32-bit or 64-bit virtual machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct GuestAddress(pub u64);
impl_address_ops!(GuestAddress, u64);

/// Type of the raw value stored in a GuestAddress object.
pub type GuestAddressValue = <GuestAddress as AddressValue>::V;

/// Type to encode offset in the guest physical address space.
pub type GuestAddressOffset = <GuestAddress as AddressValue>::V;

/// Represents a continuous region of guest physical memory.
pub trait GuestMemoryRegion: AddressRegion<A = GuestAddress, E = Error> {}

/// Represents a collection of GuestMemoryRegion objects.
///
/// The main responsibilities of the GuestMemory trait are:
/// - hide the detail of accessing guest's physical address.
/// - map a request address to a GuestMemoryRegion object and relay the request to it.
/// - handle cases where an access request spanning two or more GuestMemoryRegion objects.
pub trait GuestMemory: AddressSpace<GuestAddress, Error> {
    /// Invoke callback `f` to handle data in the address range [addr, addr + count).
    ///
    /// The address range [addr, addr + count) may span more than one AddressRegion objects, or
    /// even has holes within it. So try_access() invokes the callback 'f' for each AddressRegion
    /// object involved and returns:
    /// - error code returned by the callback 'f'
    /// - size of data already handled when encountering the first hole
    /// - size of data already handled when the whole range has been handled
    fn try_access<F>(&self, count: usize, addr: GuestAddress, mut f: F) -> Result<usize, Error>
    where
        F: FnMut(GuestAddressOffset, usize, GuestAddress, &Self::T) -> Result<usize, Error>,
    {
        //<<<<<<<<<<<<<<<<<<<<<<
        let mut cur = addr.clone();
        let mut total = 0;
        while total < count {
            match self.find_region(cur) {
                // no region for address `cur` found
                None => break,
                Some(region) => {
                    match f(total as GuestAddressOffset, count - total, cur, region) {
                        // no more data
                        Ok(0) => break,
                        // made some progress
                        Ok(len) => {
                            cur = cur
                                .checked_add(len as GuestAddressValue)
                                .ok_or(Error::InvalidGuestAddress(cur))?;
                            total += len;
                        }
                        // error happened
                        e => return e,
                    }
                }
            }
        }
        Ok(total)
        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_from() {
        let base = GuestAddress(0x100);
        let addr = GuestAddress(0x150);
        assert_eq!(addr.unchecked_offset_from(base), 0x50u64);
        assert_eq!(addr.checked_offset_from(base), Some(0x50u64));
        assert_eq!(base.checked_offset_from(addr), None);
    }

    #[test]
    fn equals() {
        let a = GuestAddress(0x300);
        let b = GuestAddress(0x300);
        let c = GuestAddress(0x301);
        assert_eq!(a, GuestAddress(a.raw_value()));
        assert_eq!(a, b);
        assert_eq!(b, a);
        assert_ne!(a, c);
        assert_ne!(c, a);
    }

    #[test]
    fn cmp() {
        let a = GuestAddress(0x300);
        let b = GuestAddress(0x301);
        assert!(a < b);
        assert!(b > a);
        assert!(!(a < a));
    }

    #[test]
    fn mask() {
        let a = GuestAddress(0x5050);
        assert_eq!(GuestAddress(0x5000), a & 0xff00u64);
        assert_eq!(GuestAddress(0x5000), a.mask(0xff00u64));
        assert_eq!(GuestAddress(0x5055), a | 0x0005u64);
    }

    #[test]
    fn add_sub() {
        let a = GuestAddress(0x50);
        let b = GuestAddress(0x60);
        assert_eq!(Some(GuestAddress(0xb0)), a.checked_add(0x60));
        assert_eq!(0x10, b.unchecked_offset_from(a));
    }

    #[test]
    fn checked_add_overflow() {
        let a = GuestAddress(0xffffffffffffff55);
        assert_eq!(Some(GuestAddress(0xffffffffffffff57)), a.checked_add(2));
        assert!(a.checked_add(0xf0).is_none());
    }

    #[test]
    fn checked_sub_underflow() {
        let a = GuestAddress(0xff);
        assert_eq!(Some(GuestAddress(0x0f)), a.checked_sub(0xf0));
        assert!(a.checked_sub(0xffff).is_none());
    }
}
