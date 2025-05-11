import _thread
import threading
import time
from datetime import datetime, timezone

from docker.models.containers import Container


# Docker uses datetimes in UTC but without the timezone info. If we pass in a tz
# then it will throw an exception.
def _utc_now_no_tz() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(tzinfo=None)


class RunningContainer:
    def __init__(
        self,
        container: Container,
        first_run: bool = False,
    ) -> None:
        self.container = container
        self.first_run = first_run
        self.running = True
        self.thread = threading.Thread(target=self._log_monitor)
        self.thread.daemon = True
        self.thread.start()

    def _log_monitor(self):
        from_date = _utc_now_no_tz() if not self.first_run else None
        to_date = _utc_now_no_tz()

        while self.running:
            try:
                for log in self.container.logs(
                    follow=False, since=from_date, until=to_date, stream=True
                ):
                    print(log.decode("utf-8"), end="")
                    # self.filter.print(log)
                time.sleep(0.1)
                from_date = to_date
                to_date = _utc_now_no_tz()
            except KeyboardInterrupt:
                print("Monitoring logs interrupted by user.")
                _thread.interrupt_main()
                break
            except Exception as e:
                print(f"Error monitoring logs: {e}")
                break

    def detach(self) -> None:
        """Stop monitoring the container logs"""
        self.running = False
        self.thread.join()

    def stop(self) -> None:
        """Stop the container"""
        self.container.stop()
        self.detach()
