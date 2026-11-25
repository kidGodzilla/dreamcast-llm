# Build:  source ../sdk/kos/environ.sh && make && make cdi
TARGET = thinking.elf
OBJS = main_dc.o dcbrain.o
KOS_CFLAGS += -O3 -ffast-math

all: rm-elf $(TARGET)

include $(KOS_BASE)/Makefile.rules

clean: rm-elf
	-rm -f $(OBJS)

rm-elf:
	-rm -f $(TARGET)

$(TARGET): $(OBJS)
	kos-cc -o $(TARGET) $(OBJS) -lm

cdi: $(TARGET)
	../sdk/mkdcdisc/builddir/mkdcdisc -q -e $(TARGET) -n "DREAMCAST THINKING" \
	  -f disc/brain.bin -f disc/tok.bin -f disc/factbook.txt -o thinking.cdi --allow-overwrite
