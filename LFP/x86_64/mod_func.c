#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _cadyn_reg(void);
extern void _capool_reg(void);
extern void _h_reg(void);
extern void _im_reg(void);
extern void _kaprox_reg(void);
extern void _kdrca1_reg(void);
extern void _kv_reg(void);
extern void _leak_reg(void);
extern void _na12_reg(void);
extern void _na16_reg(void);
extern void _na3_reg(void);
extern void _nap_reg(void);
extern void _sahp_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," cadyn.mod");
    fprintf(stderr," capool.mod");
    fprintf(stderr," h.mod");
    fprintf(stderr," im.mod");
    fprintf(stderr," kaprox.mod");
    fprintf(stderr," kdrca1.mod");
    fprintf(stderr," kv.mod");
    fprintf(stderr," leak.mod");
    fprintf(stderr," na12.mod");
    fprintf(stderr," na16.mod");
    fprintf(stderr," na3.mod");
    fprintf(stderr," nap.mod");
    fprintf(stderr," sahp.mod");
    fprintf(stderr, "\n");
  }
  _cadyn_reg();
  _capool_reg();
  _h_reg();
  _im_reg();
  _kaprox_reg();
  _kdrca1_reg();
  _kv_reg();
  _leak_reg();
  _na12_reg();
  _na16_reg();
  _na3_reg();
  _nap_reg();
  _sahp_reg();
}
