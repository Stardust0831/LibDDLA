#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

/* Landlock TCP policy inherited by the listener and every workflow step. */
struct ruleset { uint64_t fs, net; };
struct port_rule { uint64_t access, port; };
static void check(long value) {
    if (value < 0) { perror("Landlock"); exit(1); }
}
int main(int argc, char **argv) {
    if (argc < 2) return 2;
    long abi = syscall(SYS_landlock_create_ruleset, 0, 0, 1);
    if (abi < 4) { fputs("Landlock ABI 4+ required\n", stderr); return 1; }
    struct ruleset policy = {0, 3};
    int fd = syscall(SYS_landlock_create_ruleset, &policy, sizeof(policy), 0);
    check(fd);
    struct port_rule https = {2, 443};
    check(syscall(SYS_landlock_add_rule, fd, 2, &https, 0));
    check(prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0));
    check(syscall(SYS_landlock_restrict_self, fd, 0));
    close(fd);
    execvp(argv[1], argv + 1);
    perror("execvp");
    return 1;
}
