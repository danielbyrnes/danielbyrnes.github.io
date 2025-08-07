---
layout: post
title:  "Virtual Memory and Memory Management"
date:   2024-11-25 12:00:00 -0500
categories: jekyll update
---

## Illusion of Inifinite Storage

Virtual memory is a technique used by the computer operating system to create the impression that there is significantly more system memory than is physically available. Suppose a computer has 32MB or 64MB of random access memory (RAM). The system would only be able to open a handful of applications if it was only allowed to use RAM for storage. Furthermore, if the memory address space was required to be contiguous then it's possible that the system would fail to load an application even though there is technically more space available in the main memory (just not enough for the application). 

For these reasons and more, operating systems use clever techniques that utilize a combination of RAM and disk memory and a system of memory address space mappings that redirect a logical address (an address available to the user) to a physical address (an address with a physical significance on device). This system, called `Virtual Memory`, enables computers to load applications as though they have far more memory than indicated by their RAM, and allows for the more efficient use of available memory by avoiding fragmentation. `Fragmentation` happens when there isn't sufficient contiguous space to load another application, implying that the memory is inefficiently utilized.

The OS divides application memory into chunks called `pages`, which are typically on the order of magnitude of kilobytes. The pages get stored in physical memory chunks called `page frames`. A `page table` stores the mapping from the page to the page frame, and a `page fault` occures when a page is not stored in main memory and needs to be loaded from the disk. Pages should be large enough so as to minimize I/O disk accesses, but small enough so that only relevant parts of a program are being loaded to main memory.

The `MMU` is the *memory management unit*. It is usually implemented as part of the processor chip, and provides the mapping between the logical and physical memory address. The `TLB` is the *Translation Lookaside Buffer*, and is part of the MMU that stores the most recent logical to physical memory addresses. The purpose of this hardware component is to quickly look up a hardware address for pages that are most likely to be queried.

Most information here was sourced from this [virtual memory resource][virtual-mem].

[virtual-mem]: https://www.cs.umd.edu/~meesh/411/CA-online/chapter/virtual-memory-i/index.html
