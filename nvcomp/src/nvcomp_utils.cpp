#include <cuda_runtime_api.h>

#include "nvcomp.h"

nvcompStatus_t nvcompGetProperties(nvcompProperties_t *properties)
{
  if (properties == nullptr)
  {
    return nvcompErrorInvalidValue;
  }

  properties->version = NVCOMP_VER;
  properties->cudart_version = CUDART_VERSION;

  return nvcompSuccess;
}

const char *nvcompGetStatusString(nvcompStatus_t status)
{
  switch (status)
  {
    case nvcompSuccess:
      return "The operation completed successfully";
    case nvcompErrorInvalidValue:
      return "An invalid value was provided to the function";
    case nvcompErrorNotSupported:
      return "The requested operation or configuration is not supported";
    case nvcompErrorCannotDecompress:
      return "The data cannot be decompressed, possibly due to corruption or invalid format";
    case nvcompErrorBadChecksum:
      return "The checksum verification failed, indicating data corruption";
    case nvcompErrorCannotVerifyChecksums:
      return "Unable to verify checksums for the data";
    case nvcompErrorOutputBufferTooSmall:
      return "The provided output buffer size is insufficient for the operation";
    case nvcompErrorWrongHeaderLength:
      return "The header length does not match the expected value";
    case nvcompErrorAlignment:
      return "The buffer alignment does not meet the required alignment constraints";
    case nvcompErrorChunkSizeTooLarge:
      return "The chunk size exceeds the maximum allowed size";
    case nvcompErrorCannotCompress:
      return "The data cannot be compressed with the specified configuration";
    case nvcompErrorWrongInputLength:
      return "The input length is invalid or inconsistent with the operation";
    case nvcompErrorBatchSizeTooLarge:
      return "The batch size exceeds the maximum supported batch size";
    case nvcompErrorSubChunkCountTooLarge:
      return "The sub-chunk count exceeds the maximum supported per chunk";
    case nvcompErrorSubChunkCountTooSmall:
      return "The sub-chunk count is below the minimum required per chunk";
    case nvcompErrorOutputBufferAlignmentTooSmall:
      return "The output buffer alignment is too small for the data type";
    case nvcompErrorCudaError:
      return "A CUDA runtime or device error occurred during the operation";
    case nvcompErrorInternal:
      return "An internal library error occurred";
    default:
      return "Unrecognized nvcompStatus_t error code";
  }
}
