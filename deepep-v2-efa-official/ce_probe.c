/* ce_probe.c -- does this node's EFA stack support completion events (CE)?
 *
 * GDAKI's success reduces to one verb: ibv_create_comp_cntr. It needs all three
 * of a CE-capable efa.ko (>= 3.3.0, the COMP_CNTR capability bit), a libfabric
 * with the comp-cntr ABI (2.6.0amzn1.0), and rdma-core 64.0's libibverbs. Run it
 * INSIDE the container: that covers both the host kernel module and the
 * container's rdma-core in one shot -- exactly the bottom two layers of the
 * dependency chain.
 *
 * Healthy p5en.48xlarge: 16 lines, all "CE OK".
 * Driver state is PER NODE -- a freshly rebooted instance in the same fleet can
 * silently come back on an older module. Run this first whenever GDAKI worked
 * yesterday and does not today.
 *
 *   gcc -o ce_probe ce_probe.c -libverbs && ./ce_probe
 *
 * NOTE: this will not COMPILE against rdma-core < 64.0 -- ibv_comp_cntr_init_attr
 * is an incomplete type and ibv_create_comp_cntr is undeclared there. That is
 * itself a valid version check, but it means you must build it inside the 1.50.0
 * container (or on a 1.50.0 host), not on your laptop.
 */
#include <stdio.h>
#include <errno.h>
#include <infiniband/verbs.h>

int main(void) {
	int n = 0;
	struct ibv_device **d = ibv_get_device_list(&n);
	if (!d || n == 0) {
		fprintf(stderr, "no ibverbs devices -- EFA not enabled on this ENI, "
		                "or /dev/infiniband not passed into the container\n");
		return 1;
	}
	for (int i = 0; i < n; i++) {
		struct ibv_context *c = ibv_open_device(d[i]);
		if (!c) continue;
		struct ibv_comp_cntr_init_attr a = {0};
		errno = 0;
		struct ibv_comp_cntr *cc = ibv_create_comp_cntr(c, &a);
		printf("%-14s %s (errno=%d)\n", ibv_get_device_name(d[i]),
		       cc ? "CE OK" : "CE FAIL", errno);
		if (cc) ibv_destroy_comp_cntr(cc);
		ibv_close_device(c);
	}
	ibv_free_device_list(d);
	return 0;
}
