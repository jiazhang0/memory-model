// Copyright (C) 2019 Alibaba Cloud Computing. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

//! A default implementation of GuestMemory by mmap()-ing guest's memory into current process.

use libc;
use std::io::{self, Read, Write};
use std::mem;
use std::ops::{BitAnd, BitOr};
use std::os::unix::io::AsRawFd;
use std::ptr::null_mut;
use std::sync::Arc;

use address_space::{Address, AddressRegion, AddressSpace, AddressValue};
use guest_memory::*;
use volatile_memory::*;
use DataInit;

type MmapAddressValue = <MmapAddress as AddressValue>::V;
type Result<T> = std::result::Result<T, Error>;

/// Represents an offset into a memory mapped area.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct MmapAddress(pub usize);

impl_address_ops!(MmapAddress, usize);

/// A backend driver to access guest's physical memory by mmapping guest's memory into current
/// process.
/// For 32-bit hypervisor and 64-bit virtual machine, only partial of guest's physical memory maybe
/// mapped into current process due to limited process virtual address space size.
#[derive(Debug)]
pub struct MmapRegion {
    addr: *mut u8,
    size: usize,
}

// Send and Sync aren't automatically inherited for the raw address pointer.
// Accessing that pointer is only done through the stateless interface which
// allows the object to be shared by multiple threads without a decrease in
// safety.
unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

impl MmapRegion {
    /// Creates an anonymous shared mapping of `size` bytes.
    ///
    /// # Arguments
    /// * `size` - Size of memory region in bytes.
    pub fn new(size: usize) -> Result<Self> {
        // This is safe because we are creating an anonymous mapping in a place not already used by
        // any other area in this process.
        let addr = unsafe {
            libc::mmap(
                null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_SHARED | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(Error::SystemCallFailed(io::Error::last_os_error()));
        }
        Ok(Self {
            addr: addr as *mut u8,
            size,
        })
    }

    /// Maps the `size` bytes starting at `offset` bytes of the given `fd`.
    ///
    /// # Arguments
    /// * `fd` - File descriptor to mmap from.
    /// * `size` - Size of memory region in bytes.
    /// * `offset` - Offset in bytes from the beginning of `fd` to start the mmap.
    pub fn from_fd(fd: &AsRawFd, size: usize, offset: usize) -> Result<Self> {
        if offset > libc::off_t::max_value() as usize {
            return Err(Error::InvalidBackendOffset);
        }
        // This is safe because we are creating a mapping in a place not already used by any other
        // area in this process.
        let addr = unsafe {
            libc::mmap(
                null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                offset as libc::off_t,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(Error::SystemCallFailed(io::Error::last_os_error()));
        }
        Ok(Self {
            addr: addr as *mut u8,
            size,
        })
    }

    /// Returns a pointer to the beginning of the memory region.  Should only be
    /// used for passing this region to ioctls for setting guest memory.
    pub fn as_ptr(&self) -> *mut u8 {
        self.addr
    }

    unsafe fn as_slice(&self) -> &[u8] {
        // This is safe because we mapped the area at addr ourselves, so this slice will not
        // overflow. However, it is possible to alias.
        std::slice::from_raw_parts(self.addr, self.size)
    }

    unsafe fn as_mut_slice(&self) -> &mut [u8] {
        // This is safe because we mapped the area at addr ourselves, so this slice will not
        // overflow. However, it is possible to alias.
        std::slice::from_raw_parts_mut(self.addr, self.size)
    }

    // Check that addr + count is valid and return the sum.
    fn region_end(&self, addr: usize, count: usize) -> Result<usize> {
        let end = addr
            .checked_add(count)
            .ok_or(Error::InvalidBackendAddress)?;
        if end > self.size {
            return Err(Error::InvalidBackendAddress);
        }
        Ok(end)
    }
}

impl AddressRegion for MmapRegion {
    type A = MmapAddress;
    type E = Error;

    fn size(&self) -> MmapAddressValue {
        self.size
    }

    fn max_addr(&self) -> MmapAddress {
        MmapAddress(self.size)
    }

    fn is_valid(&self) -> bool {
        !self.addr.is_null() && self.addr != libc::MAP_FAILED as *mut u8
    }

    /// Writes a slice to the region at the specified address.
    /// Returns the number of bytes written. The number of bytes written can
    /// be less than the length of the slice if there isn't enough room in the
    /// region.
    ///
    /// # Examples
    /// * Write a slice at offset 256.
    ///
    /// ```
    /// #   use memory_model::{AddressRegion, MmapAddress, MmapRegion};
    /// #   let mut mem_map = MmapRegion::new(1024).unwrap();
    ///     let res = mem_map.write(&[1,2,3,4,5], MmapAddress(1020));
    ///     assert!(res.is_ok());
    ///     assert_eq!(res.unwrap(), 4);
    /// ```
    fn write(&self, buf: &[u8], addr: MmapAddress) -> Result<usize> {
        if addr.raw_value() >= self.size {
            return Err(Error::InvalidBackendAddress);
        }
        unsafe {
            // Guest memory can't strictly be modeled as a slice because it is
            // volatile.  Writing to it with what compiles down to a memcpy
            // won't hurt anything as long as we get the bounds checks right.
            let mut slice: &mut [u8] = &mut self.as_mut_slice()[addr.raw_value()..];
            Ok(slice.write(buf).map_err(Error::WriteToMemory)?)
        }
    }

    /// Reads to a slice from the region at the specified address.
    /// Returns the number of bytes read. The number of bytes read can be less than the length
    /// of the slice if there isn't enough room in the region.
    ///
    /// # Examples
    /// * Read a slice of size 16 at offset 256.
    ///
    /// ```
    /// #   use memory_model::{AddressRegion, MmapAddress, MmapRegion};
    /// #   let mut mem_map = MmapRegion::new(1024).unwrap();
    ///     let buf = &mut [0u8; 16];
    ///     let res = mem_map.read(buf, MmapAddress(1010));
    ///     assert!(res.is_ok());
    ///     assert_eq!(res.unwrap(), 14);
    /// ```
    fn read(&self, mut buf: &mut [u8], addr: MmapAddress) -> Result<usize> {
        if addr.raw_value() >= self.size {
            return Err(Error::InvalidBackendAddress);
        }
        unsafe {
            // Guest memory can't strictly be modeled as a slice because it is
            // volatile.  Writing to it with what compiles down to a memcpy
            // won't hurt anything as long as we get the bounds checks right.
            let slice: &[u8] = &self.as_slice()[addr.raw_value()..];
            Ok(buf.write(slice).map_err(Error::ReadFromMemory)?)
        }
    }

    /// Writes a slice to the region at the specified address.
    ///
    /// # Examples
    /// * Write a slice at offset 256.
    ///
    /// ```
    /// #   use memory_model::{AddressRegion, MmapAddress, MmapRegion};
    /// #   let mut mem_map = MmapRegion::new(1024).unwrap();
    ///     let res = mem_map.write_slice(&[1,2,3,4,5], MmapAddress(256));
    ///     assert!(res.is_ok());
    ///     assert_eq!(res.unwrap(), ());
    /// ```
    fn write_slice(&self, buf: &[u8], addr: MmapAddress) -> Result<()> {
        let len = self.write(buf, addr)?;
        if len != buf.len() {
            return Err(Error::ShortWrite {
                expected: buf.len() as u64,
                completed: len as u64,
            });
        }
        Ok(())
    }

    /// Reads to a slice from the region at the specified address.
    ///
    /// # Examples
    /// * Read a slice of size 16 at offset 256.
    ///
    /// ```
    /// #   use memory_model::{AddressRegion, MmapAddress, MmapRegion};
    /// #   let mut mem_map = MmapRegion::new(1024).unwrap();
    ///     let buf = &mut [0u8; 16];
    ///     let res = mem_map.read_slice(buf, MmapAddress(256));
    ///     assert!(res.is_ok());
    ///     assert_eq!(res.unwrap(), ());
    /// ```
    fn read_slice(&self, buf: &mut [u8], addr: MmapAddress) -> Result<()> {
        let len = self.read(buf, addr)?;
        if len != buf.len() {
            return Err(Error::ShortRead {
                expected: buf.len() as u64,
                completed: len as u64,
            });
        }
        Ok(())
    }

    /// Writes an object to the region at the specified address.
    /// Returns Ok(()) if the object fits, or Err if it extends past the end.
    ///
    /// # Examples
    /// * Write a u64 at offset 16.
    ///
    /// ```
    /// #   use memory_model::{AddressRegion, MmapAddress, MmapRegion};
    /// #   let mut mem_map = MmapRegion::new(1024).unwrap();
    ///     let res = mem_map.write_obj(55u64, MmapAddress(16));
    ///     assert!(res.is_ok());
    /// ```
    fn write_obj<T: DataInit>(&self, val: T, addr: MmapAddress) -> Result<()> {
        unsafe {
            // Guest memory can't strictly be modeled as a slice because it is
            // volatile.  Writing to it with what compiles down to a memcpy
            // won't hurt anything as long as we get the bounds checks right.
            self.region_end(addr.raw_value(), mem::size_of::<T>())?;
            std::ptr::write_volatile(
                &mut self.as_mut_slice()[addr.raw_value()..] as *mut _ as *mut T,
                val,
            );
            Ok(())
        }
    }

    /// Reads an object from the region at the given address.
    /// Reading from a volatile area isn't strictly safe as it could change mid-read.
    /// However, as long as the type T is plain old data and can handle random initialization,
    /// everything will be OK.
    ///
    /// # Examples
    /// * Read a u64 written to offset 32.
    ///
    /// ```
    /// #   use memory_model::{AddressRegion, MmapAddress, MmapRegion};
    /// #   let mut mem_map = MmapRegion::new(1024).unwrap();
    ///     let res = mem_map.write_obj(55u64, MmapAddress(32));
    ///     assert!(res.is_ok());
    ///     let num: u64 = mem_map.read_obj(MmapAddress(32)).unwrap();
    ///     assert_eq!(55, num);
    /// ```
    fn read_obj<T: DataInit>(&self, addr: MmapAddress) -> Result<T> {
        self.region_end(addr.raw_value(), mem::size_of::<T>())?;
        unsafe {
            // This is safe because by definition Copy types can have their bits
            // set arbitrarily and still be valid.
            Ok(std::ptr::read_volatile(
                &self.as_slice()[addr.raw_value()..] as *const _ as *const T,
            ))
        }
    }

    /// Writes data from a readable object like a File and writes it to the region.
    ///
    /// # Examples
    ///
    /// * Read bytes from /dev/urandom
    ///
    /// ```
    /// # use memory_model::{AddressRegion, MmapAddress, MmapRegion};
    /// # use std::fs::File;
    /// # use std::path::Path;
    /// # fn test_read_random() -> Result<u32, ()> {
    /// #     let mut mem_map = MmapRegion::new(1024).unwrap();
    ///       let mut file = File::open(Path::new("/dev/urandom")).map_err(|_| ())?;
    ///       mem_map.write_from_stream(MmapAddress(32), &mut file, 128).map_err(|_| ())?;
    ///       let rand_val: u32 =  mem_map.read_obj(MmapAddress(40)).map_err(|_| ())?;
    /// #     Ok(rand_val)
    /// # }
    /// ```
    fn write_from_stream<F>(&self, addr: MmapAddress, src: &mut F, count: usize) -> Result<()>
    where
        F: Read,
    {
        let end = self.region_end(addr.raw_value(), count)?;
        unsafe {
            // It is safe to overwrite the volatile memory. Accessing the guest
            // memory as a mutable slice is OK because nothing assumes another
            // thread won't change what is loaded.
            let dst = &mut self.as_mut_slice()[addr.raw_value()..end];
            src.read_exact(dst).map_err(Error::ReadFromSource)?;
        }
        Ok(())
    }

    /// Reads data from the region to a writable object.
    ///
    /// # Examples
    ///
    /// * Write 128 bytes to /dev/null
    ///
    /// ```
    /// # use memory_model::{AddressRegion, MmapAddress, MmapRegion};
    /// # use std::fs::File;
    /// # use std::path::Path;
    /// # fn test_write_null() -> Result<(), ()> {
    /// #     let mut mem_map = MmapRegion::new(1024).unwrap();
    ///       let mut file = File::open(Path::new("/dev/null")).map_err(|_| ())?;
    ///       mem_map.read_into_stream(MmapAddress(32), &mut file, 128).map_err(|_| ())?;
    /// #     Ok(())
    /// # }
    /// ```
    fn read_into_stream<F>(&self, addr: MmapAddress, dst: &mut F, count: usize) -> Result<()>
    where
        F: Write,
    {
        let end = self.region_end(addr.raw_value(), count)?;
        unsafe {
            // It is safe to read from volatile memory. Accessing the guest
            // memory as a slice is OK because nothing assumes another thread
            // won't change what is loaded.
            let src = &self.as_mut_slice()[addr.raw_value()..end];
            dst.write_all(src).map_err(Error::ReadFromSource)?;
        }
        Ok(())
    }
}

impl VolatileMemory for MmapRegion {
    fn get_slice(&self, offset: usize, count: usize) -> VolatileMemoryResult<VolatileSlice> {
        let end = calc_offset(offset, count)?;
        if end > self.size {
            return Err(VolatileMemoryError::OutOfBounds { addr: end });
        }

        // Safe because we checked that offset + count was within our range and we only ever hand
        // out volatile accessors.
        Ok(unsafe { VolatileSlice::new((self.addr as usize + offset) as *mut _, count) })
    }
}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        // This is safe because we mmap the area at addr ourselves, and nobody
        // else is holding a reference to it.
        unsafe {
            libc::munmap(self.addr as *mut libc::c_void, self.size);
        }
    }
}

/// Tracks a mapping of memory in the current process and the corresponding base address
/// in the guest's memory space.
pub struct GuestRegionMmap {
    mapping: MmapRegion,
    guest_base: GuestAddress,
}

impl GuestRegionMmap {
    /// Create a new memory-mapped memory region for guest's physical memory.
    /// Note: caller needs to ensure that (mapping.size() + guest_base) doesn't wrapping around.
    pub fn new(mapping: MmapRegion, guest_base: GuestAddress) -> Self {
        GuestRegionMmap {
            mapping,
            guest_base,
        }
    }

    fn to_mmap_addr(&self, addr: GuestAddress) -> Result<MmapAddress> {
        let offset = addr
            .checked_offset_from(self.guest_base)
            .ok_or(Error::InvalidGuestAddress(addr))?;
        if offset >= self.size() {
            return Err(Error::InvalidGuestAddress(addr));
        }
        Ok(MmapAddress(offset as usize))
    }

    unsafe fn as_slice(&self) -> &[u8] {
        self.mapping.as_slice()
    }

    unsafe fn as_mut_slice(&self) -> &mut [u8] {
        self.mapping.as_mut_slice()
    }
}

impl AddressRegion for GuestRegionMmap {
    type A = GuestAddress;
    type E = Error;

    fn size(&self) -> GuestAddressValue {
        self.mapping.size() as GuestAddressValue
    }

    fn min_addr(&self) -> GuestAddress {
        self.guest_base
    }

    fn max_addr(&self) -> GuestAddress {
        // unchecked_add is safe as the region bounds were checked when it was created.
        self.guest_base
            .unchecked_add(self.mapping.size() as GuestAddressValue)
    }

    fn write(&self, buf: &[u8], addr: GuestAddress) -> Result<usize> {
        let maddr = self.to_mmap_addr(addr)?;
        self.mapping.write(buf, maddr)
    }

    fn read(&self, buf: &mut [u8], addr: GuestAddress) -> Result<usize> {
        let maddr = self.to_mmap_addr(addr)?;
        self.mapping.read(buf, maddr)
    }

    fn write_slice(&self, buf: &[u8], addr: GuestAddress) -> Result<()> {
        let maddr = self.to_mmap_addr(addr)?;
        self.mapping.write_slice(buf, maddr)
    }

    fn read_slice(&self, buf: &mut [u8], addr: GuestAddress) -> Result<()> {
        let maddr = self.to_mmap_addr(addr)?;
        self.mapping.read_slice(buf, maddr)
    }

    fn write_obj<T: DataInit>(&self, val: T, addr: GuestAddress) -> Result<()> {
        let maddr = self.to_mmap_addr(addr)?;
        self.mapping.write_obj::<T>(val, maddr)
    }

    fn read_obj<T: DataInit>(&self, addr: GuestAddress) -> Result<T> {
        let maddr = self.to_mmap_addr(addr)?;
        self.mapping.read_obj::<T>(maddr)
    }

    fn write_from_stream<F>(&self, addr: GuestAddress, src: &mut F, count: usize) -> Result<()>
    where
        F: Read,
    {
        let maddr = self.to_mmap_addr(addr)?;
        self.mapping.write_from_stream::<F>(maddr, src, count)
    }

    fn read_into_stream<F>(&self, addr: GuestAddress, dst: &mut F, count: usize) -> Result<()>
    where
        F: Write,
    {
        let maddr = self.to_mmap_addr(addr)?;
        self.mapping.read_into_stream::<F>(maddr, dst, count)
    }
}

impl GuestMemoryRegion for GuestRegionMmap {}

/// Tracks memory regions allocated/mapped for the guest in the current process.
#[derive(Clone)]
struct GuestMemoryMmap {
    regions: Arc<Vec<GuestRegionMmap>>,
}

impl GuestMemoryMmap {}

impl AddressRegion for GuestMemoryMmap {
    type A = GuestAddress;
    type E = Error;

    fn size(&self) -> GuestAddressValue {
        self.regions
            .iter()
            .map(|region| region.mapping.size() as GuestAddressValue)
            .sum()
    }

    fn min_addr(&self) -> GuestAddress {
        self.regions
            .iter()
            .min_by_key(|region| region.min_addr())
            .map_or(GuestAddress(0), |region| region.min_addr())
    }

    fn max_addr(&self) -> GuestAddress {
        self.regions
            .iter()
            .max_by_key(|region| region.max_addr())
            .map_or(GuestAddress(0), |region| region.max_addr())
    }

    fn is_valid(&self) -> bool {
        // TODO: verify there's no intersection among regions
        true
    }

    fn address_in_range(&self, addr: GuestAddress) -> bool {
        for region in self.regions.iter() {
            if addr >= region.min_addr() && addr < region.max_addr() {
                return true;
            }
        }
        false
    }

    fn write(&self, buf: &[u8], addr: GuestAddress) -> Result<usize> {
        self.try_access(
            buf.len(),
            addr,
            |offset, _count, caddr, region| -> Result<usize> {
                if offset >= buf.len() as GuestAddressValue {
                    return Err(Error::InvalidBackendOffset);
                }
                region.write(&buf[offset as usize..], caddr)
            },
        )
    }

    fn read(&self, buf: &mut [u8], addr: GuestAddress) -> Result<usize> {
        self.try_access(
            buf.len(),
            addr,
            |offset, _count, caddr, region| -> Result<usize> {
                if offset >= buf.len() as GuestAddressValue {
                    return Err(Error::InvalidBackendOffset);
                }
                region.read(&mut buf[offset as usize..], caddr)
            },
        )
    }

    fn write_slice(&self, buf: &[u8], addr: GuestAddress) -> Result<()> {
        let res = self.try_access(
            buf.len(),
            addr,
            |offset, _count, caddr, region| -> Result<usize> {
                if offset >= buf.len() as GuestAddressValue {
                    return Err(Error::InvalidBackendOffset);
                }
                region.write(&buf[offset as usize..], caddr)
            },
        )?;
        if res != buf.len() {
            return Err(Error::ShortWrite {
                expected: buf.len() as GuestAddressValue,
                completed: res as GuestAddressValue,
            });
        }
        Ok(())
    }

    fn read_slice(&self, buf: &mut [u8], addr: GuestAddress) -> Result<()> {
        let res = self.try_access(
            buf.len(),
            addr,
            |offset, _count, caddr, region| -> Result<usize> {
                if offset >= buf.len() as GuestAddressValue {
                    return Err(Error::InvalidBackendOffset);
                }
                region.read(&mut buf[offset as usize..], caddr)
            },
        )?;
        if res != buf.len() {
            return Err(Error::ShortRead {
                expected: buf.len() as GuestAddressValue,
                completed: res as GuestAddressValue,
            });
        }
        Ok(())
    }

    fn write_obj<T: DataInit>(&self, _val: T, _addr: GuestAddress) -> Result<()> {
        // Do we really need to support write_volatile acrossing region boundary?
        Err(Error::InvalidBackendOperation)
    }

    fn read_obj<T: DataInit>(&self, _addr: GuestAddress) -> Result<T> {
        // Do we really need to support read_volatile acrossing region boundary?
        Err(Error::InvalidBackendOperation)
    }

    fn write_from_stream<F>(&self, addr: GuestAddress, src: &mut F, count: usize) -> Result<()>
    where
        F: Read,
    {
        let res = self.try_access(count, addr, |offset, cnt, caddr, region| -> Result<usize> {
            // Something bad happened...
            if offset >= count as GuestAddressValue {
                return Err(Error::InvalidBackendOffset);
            }
            // This is safe cauase the `caddr` is within the `region`.
            let start = caddr.unchecked_offset_from(region.min_addr()) as usize;
            let cap = region.max_addr().unchecked_offset_from(caddr) as usize;
            let len = std::cmp::min(cap, cnt);
            let end = start + len;
            let dst = unsafe { &mut region.as_mut_slice()[start..end] };
            src.read_exact(dst).map_err(Error::ReadFromSource)?;
            Ok(len)
        })?;
        if res != count {
            return Err(Error::ShortWrite {
                expected: count as GuestAddressValue,
                completed: res as GuestAddressValue,
            });
        }
        Ok(())
    }

    /// Reads data from the region to a writable object.
    ///
    /// # Examples
    ///
    /// * Write 128 bytes to /dev/null
    ///
    /// ```
    /// # use memory_model::{AddressRegion, MmapAddress, MmapRegion};
    /// # use std::fs::File;
    /// # use std::path::Path;
    /// # fn test_write_null() -> Result<(), ()> {
    /// #     let mut mem_map = MmapRegion::new(1024).unwrap();
    ///       let mut file = File::open(Path::new("/dev/null")).map_err(|_| ())?;
    ///       mem_map.read_into_stream(MmapAddress(32), &mut file, 128).map_err(|_| ())?;
    /// #     Ok(())
    /// # }
    /// ```
    fn read_into_stream<F>(&self, addr: GuestAddress, dst: &mut F, count: usize) -> Result<()>
    where
        F: Write,
    {
        let res = self.try_access(count, addr, |offset, cnt, caddr, region| -> Result<usize> {
            // Something bad happened...
            if offset >= count as GuestAddressValue {
                return Err(Error::InvalidBackendOffset);
            }
            // This is safe cauase the `caddr` is within the `region`.
            let start = caddr.unchecked_offset_from(region.min_addr()) as usize;
            let cap = region.max_addr().unchecked_offset_from(caddr) as usize;
            let len = std::cmp::min(cap, cnt);
            let end = start + len;
            let src = unsafe { &region.as_slice()[start..end] };
            // It is safe to read from volatile memory. Accessing the guest
            // memory as a slice is OK because nothing assumes another thread
            // won't change what is loaded.
            dst.write_all(src).map_err(Error::ReadFromSource)?;
            Ok(len)
        })?;
        if res != count {
            return Err(Error::ShortRead {
                expected: count as GuestAddressValue,
                completed: res as GuestAddressValue,
            });
        }
        Ok(())
    }
}

impl AddressSpace<GuestAddress, Error> for GuestMemoryMmap {
    type T = GuestRegionMmap;

    fn num_regions(&self) -> usize {
        self.regions.len()
    }

    fn find_region(&self, addr: GuestAddress) -> Option<&Self::T> {
        for ref region in self.regions.iter() {
            if addr >= region.min_addr() && addr < region.max_addr() {
                return Some(region);
            }
        }
        None
    }

    fn with_regions<F>(&self, cb: F) -> Result<()>
    where
        F: Fn(usize, &Self::T) -> Result<()>,
    {
        for (index, ref region) in self.regions.iter().enumerate() {
            cb(index, region)?;
        }
        Ok(())
    }

    fn with_regions_mut<F>(&self, mut cb: F) -> Result<()>
    where
        F: FnMut(usize, &Self::T) -> Result<()>,
    {
        for (index, ref region) in self.regions.iter().enumerate() {
            cb(index, region)?;
        }
        Ok(())
    }
}

impl GuestMemory for GuestMemoryMmap {}

#[cfg(test)]
mod tests {
    extern crate tempfile;

    use self::tempfile::tempfile;
    use super::*;
    use std::fs::File;
    use std::mem;
    use std::os::unix::io::FromRawFd;
    use std::path::Path;

    #[test]
    fn basic_map() {
        let m = MmapRegion::new(1024).unwrap();
        assert_eq!(1024, m.size());
    }

    #[test]
    fn map_invalid_size() {
        let res = MmapRegion::new(0).unwrap_err();
        if let Error::SystemCallFailed(e) = res {
            assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
        } else {
            panic!("unexpected error: {:?}", res);
        }
    }

    #[test]
    fn map_invalid_fd() {
        let fd = unsafe { std::fs::File::from_raw_fd(-1) };
        let res = MmapRegion::from_fd(&fd, 1024, 0).unwrap_err();
        if let Error::SystemCallFailed(e) = res {
            assert_eq!(e.raw_os_error(), Some(libc::EBADF));
        } else {
            panic!("unexpected error: {:?}", res);
        }
    }

    #[test]
    fn slice_size() {
        let m = MmapRegion::new(5).unwrap();
        let s = m.get_slice(2, 3).unwrap();
        assert_eq!(s.size(), 3);
    }

    #[test]
    fn slice_addr() {
        let m = MmapRegion::new(5).unwrap();
        let s = m.get_slice(2, 3).unwrap();
        assert_eq!(s.as_ptr(), unsafe { m.as_ptr().offset(2) });
    }

    #[test]
    fn slice_store() {
        let m = MmapRegion::new(5).unwrap();
        let r = m.get_ref(2).unwrap();
        r.store(9u16);
        assert_eq!(m.read_obj::<u16>(MmapAddress(2)).unwrap(), 9);
    }

    #[test]
    fn slice_overflow_error() {
        let m = MmapRegion::new(5).unwrap();
        let res = m.get_slice(std::usize::MAX, 3).unwrap_err();
        assert_eq!(
            res,
            VolatileMemoryError::Overflow {
                base: std::usize::MAX,
                offset: 3,
            }
        );
    }

    #[test]
    fn slice_oob_error() {
        let m = MmapRegion::new(5).unwrap();
        let res = m.get_slice(3, 3).unwrap_err();
        assert_eq!(res, VolatileMemoryError::OutOfBounds { addr: 6 });
    }

    #[test]
    fn from_fd_offset_invalid() {
        let fd = unsafe { std::fs::File::from_raw_fd(-1) };
        let res =
            MmapRegion::from_fd(&fd, 4096, (libc::off_t::max_value() as usize) + 1).unwrap_err();
        match res {
            Error::InvalidBackendOffset => {}
            e => panic!("unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_write_past_end() {
        let m = MmapRegion::new(5).unwrap();
        let res = m.write(&[1, 2, 3, 4, 5, 6], MmapAddress(0));
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), 5);
    }

    #[test]
    fn slice_read_and_write() {
        let mem_map = MmapRegion::new(5).unwrap();
        let sample_buf = [1, 2, 3];
        assert!(mem_map.write(&sample_buf, MmapAddress(5)).is_err());
        assert!(mem_map.write(&sample_buf, MmapAddress(2)).is_ok());
        let mut buf = [0u8; 3];
        assert!(mem_map.read(&mut buf, MmapAddress(5)).is_err());
        assert!(mem_map.read_slice(&mut buf, MmapAddress(2)).is_ok());
        assert_eq!(buf, sample_buf);
    }

    #[test]
    fn obj_read_and_write() {
        let mem_map = MmapRegion::new(5).unwrap();
        assert!(mem_map.write_obj(55u16, MmapAddress(4)).is_err());
        assert!(mem_map
            .write_obj(55u16, MmapAddress(core::usize::MAX))
            .is_err());
        assert!(mem_map.write_obj(55u16, MmapAddress(2)).is_ok());
        assert_eq!(mem_map.read_obj::<u16>(MmapAddress(2)).unwrap(), 55u16);
        assert!(mem_map.read_obj::<u16>(MmapAddress(4)).is_err());
        assert!(mem_map
            .read_obj::<u16>(MmapAddress(core::usize::MAX))
            .is_err());
    }

    #[test]
    fn mem_read_and_write() {
        let mem_map = MmapRegion::new(5).unwrap();
        assert!(mem_map.write_obj(!0u32, MmapAddress(1)).is_ok());
        let mut file = File::open(Path::new("/dev/zero")).unwrap();
        assert!(mem_map
            .write_from_stream(MmapAddress(2), &mut file, mem::size_of::<u32>())
            .is_err());
        assert!(mem_map
            .write_from_stream(
                MmapAddress(core::usize::MAX),
                &mut file,
                mem::size_of::<u32>()
            )
            .is_err());

        assert!(mem_map
            .write_from_stream(MmapAddress(1), &mut file, mem::size_of::<u32>())
            .is_ok());

        let mut f = tempfile().unwrap();
        assert!(mem_map
            .write_from_stream(MmapAddress(1), &mut f, mem::size_of::<u32>())
            .is_err());
        format!(
            "{:?}",
            mem_map.write_from_stream(MmapAddress(1), &mut f, mem::size_of::<u32>())
        );

        assert_eq!(mem_map.read_obj::<u32>(MmapAddress(1)).unwrap(), 0);

        let mut sink = Vec::new();
        assert!(mem_map
            .read_into_stream(MmapAddress(1), &mut sink, mem::size_of::<u32>())
            .is_ok());
        assert!(mem_map
            .read_into_stream(MmapAddress(2), &mut sink, mem::size_of::<u32>())
            .is_err());
        assert!(mem_map
            .read_into_stream(
                MmapAddress(core::usize::MAX),
                &mut sink,
                mem::size_of::<u32>()
            )
            .is_err());
        format!(
            "{:?}",
            mem_map.read_into_stream(MmapAddress(2), &mut sink, mem::size_of::<u32>())
        );
        assert_eq!(sink, vec![0; mem::size_of::<u32>()]);
    }

    #[test]
    fn mapped_file_read() {
        let mut f = tempfile().unwrap();
        let sample_buf = &[1, 2, 3, 4, 5];
        assert!(f.write_all(sample_buf).is_ok());

        let mem_map = MmapRegion::from_fd(&f, sample_buf.len(), 0).unwrap();
        let buf = &mut [0u8; 16];
        assert_eq!(mem_map.read(buf, MmapAddress(0)).unwrap(), sample_buf.len());
        assert_eq!(buf[0..sample_buf.len()], sample_buf[..]);
    }
}
