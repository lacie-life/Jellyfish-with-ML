CUFLAGS  := $(CUFLAGS) -std=c++11
CUFLAGSD := $(CUFLAGSD) -std=c++11 -g -G
OUTNAME_RELEASE = sample_nmt
OUTNAME_DEBUG   = sample_nmt_debug
EXTRA_DIRECTORIES = ../common data model
SAMPLE_DIR_NAME = $(shell basename $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
MAKEFILE ?= ../Makefile.config
include $(MAKEFILE)
INCPATHS += -I"."
