#include "api_grid_test_common.h"

using namespace api_grid_test;

// ddlaMalloc / ddlaFree on the handle, typed and void* forms.
void check_device_memory(const ddla::DdlaHandle_t& handle, const Shape&)
{
    double host_value = 0.0;

    double* typed_zero = &host_value;
    DDLA_CHECK(ddlaMalloc(handle, &typed_zero, 0));
    require_close(handle, "ddlaMalloc(typed, zero)",
                  typed_zero == nullptr ? 0.0 : 1.0, 0.0);
    DDLA_CHECK(ddlaFree(handle, typed_zero));

    void* raw_zero = &host_value;
    DDLA_CHECK(ddlaMalloc(handle, &raw_zero, 0));
    require_close(handle, "ddlaMalloc(void, zero)",
                  raw_zero == nullptr ? 0.0 : 1.0, 0.0);
    DDLA_CHECK(ddlaFree(handle, raw_zero));

    DDLA_CHECK(ddlaFree(handle, nullptr));

    const ddla::ddlaStatus_t null_output_status = ddlaMalloc(handle, nullptr, 0);
    require_close(handle, "ddlaMalloc(null output)",
                  null_output_status != ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS ? 0.0 : 1.0, 0.0);
    DDLA_CHECK(ddlaGetLastError(handle));

    double* d_typed = nullptr;
    DDLA_CHECK(ddlaMalloc(handle, &d_typed, sizeof(double)));
    require_close(handle, "ddlaMalloc(typed, nonzero)",
                  d_typed != nullptr ? 0.0 : 1.0, 0.0);
    DDLA_CHECK(ddlaFree(handle, d_typed));

    void* d_raw = nullptr;
    DDLA_CHECK(ddlaMalloc(handle, &d_raw, sizeof(double)));
    require_close(handle, "ddlaMalloc(void, nonzero)",
                  d_raw != nullptr ? 0.0 : 1.0, 0.0);
    DDLA_CHECK(ddlaFree(handle, d_raw));
    check_ddla_sync(handle);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_device_memory", check_device_memory);
}
